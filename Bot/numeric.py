import datetime
import numpy as np
import asyncio
import re
import numpy as np
from prompts import (
    NUMERIC_PROMPT_historical,
    NUMERIC_PROMPT_current,
    NUMERIC_PROMPT_1,
    NUMERIC_PROMPT_2,
    NUMERIC_PROMPT_MONTE_CARLO,
)
from llm_calls import call_claude, call_gpt, extract_and_run_python_code
from search import process_search_queries

def generate_continuous_cdf(
    percentile_values: dict[float, float],
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
) -> list[float]:
    import numpy as np

    # Ensure float values
    percentile_values = {float(k): float(v) for k, v in percentile_values.items()}

    # Add extrapolated endpoints
    def extrapolate(points: list[tuple[float, float]], target_p: float) -> float:
        x = [p for p, _ in points]
        y = [v for _, v in points]
        coeffs = np.polyfit(x, y, 1)
        return np.polyval(coeffs, target_p)

    keys_sorted = sorted(percentile_values.keys())
    if not open_lower_bound:
        percentile_values[0.0] = lower_bound
    else:
        if len(keys_sorted) >= 2:
            points = sorted(percentile_values.items())[:2]
            percentile_values[0.0] = extrapolate(points, 0.0)
        else:
            raise ValueError("Not enough points to extrapolate lower bound.")

    if not open_upper_bound:
        percentile_values[100.0] = upper_bound
    else:
        if len(keys_sorted) >= 2:
            points = sorted(percentile_values.items())[-2:]
            percentile_values[100.0] = extrapolate(points, 100.0)
        else:
            raise ValueError("Not enough points to extrapolate upper bound.")

    # Now sort by percentile (not value!)
    sorted_items = sorted(percentile_values.items(), key=lambda x: x[0])
    percentiles = np.array([p for p, _ in sorted_items]) / 100.0
    values = np.array([v for _, v in sorted_items])

    # Ensure values are non-decreasing
    if not np.all(np.diff(values) >= 0):
        raise ValueError("Percentile values must be non-decreasing.")

    # Create uniform x-axis across full bounds (Metaculus expects this)
    if zero_point is None:
        cdf_x = np.linspace(lower_bound, upper_bound, 201)
    else:
        # Optional log-symmetric x-axis (rarely needed)
        left_span = max(zero_point - lower_bound, 1e-8)
        right_span = max(upper_bound - zero_point, 1e-8)
        left = zero_point - np.logspace(np.log10(left_span), 0, 101)[::-1]
        right = zero_point + np.logspace(0, np.log10(right_span), 100)
        cdf_x = np.unique(np.concatenate([left, [zero_point], right]))
        cdf_x = np.clip(cdf_x, lower_bound, upper_bound)

    # Interpolate to get CDF values
    cdf_y = np.interp(cdf_x, values, percentiles, left=0.0, right=1.0)

    # Handle open bounds
    if open_lower_bound:
        cdf_y = np.clip(cdf_y, 0.001, 1.0)
    else:
        cdf_y[0] = 0.0
        cdf_y = np.clip(cdf_y, 0.0, 1.0)

    if open_upper_bound:
        cdf_y = np.clip(cdf_y, 0.0, 0.999)
    else:
        cdf_y[-1] = 1.0

    # Enforce Metaculus slope rule: minimum step size
    min_step = 5e-5
    for i in range(1, len(cdf_y)):
        if cdf_y[i] - cdf_y[i - 1] < min_step:
            cdf_y[i] = cdf_y[i - 1] + min_step
    cdf_y = np.clip(cdf_y, 0.0, 1.0)
    if not open_upper_bound:
        cdf_y[-1] = 1.0

    # Final validation
    if not np.all(np.diff(cdf_y) >= 0):
        raise RuntimeError("Generated CDF is non-monotonic")

    return cdf_y.tolist()

def extract_percentiles_from_response(forecast_text: str, verbose=True) -> dict:
    """
    Extracts a dict of percentiles from a forecast text block.
    Starts extraction only after a line containing 'Distribution:'.
    """
    percentiles = {}
    collecting = False

    VALID_KEYS = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                  50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99}

    # Match things like "Percentile 10: 12.5%" or "10: 13.0"
    pattern = re.compile(
        r"^(?:Percentile\s+)?(\d{1,3})\s*[:\-]\s*(-?\d+(?:\.\d+)?)(?:%?)\s*$",
        re.IGNORECASE
    )

    lines = forecast_text.splitlines()

    if verbose:
        print("\n🔍 Starting extraction after 'Distribution:' anchor...")
        print("----------- Raw Forecast Text -----------")
        print(forecast_text[:1000])
        print("-----------------------------------------\n")

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not collecting and "distribution:" in stripped.lower():
            collecting = True
            if verbose:
                print(f"🚩 Found 'Distribution:' anchor at line {i + 1}")
            continue

        if not collecting:
            if verbose:
                print(f"⏭️ Skipping pre-distribution line {i + 1}: {stripped}")
            continue

        if verbose:
            print(f"📄 Line {i + 1}: '{stripped}'")

        match = pattern.match(stripped)
        if match:
            p_raw, val_raw = match.groups()
            try:
                p = int(p_raw)
                val = float(val_raw)
                if p in VALID_KEYS:
                    percentiles[p] = val
                    if verbose:
                        print(f"✅ Matched Percentile {p}: {val}")
                else:
                    if verbose:
                        print(f"⚠️ Ignored unrecognized percentile key: {p}")
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to parse '{p_raw}: {val_raw}' -> {e}")
        else:
            if verbose:
                print("⛔ No match")

    if not percentiles:
        raise ValueError(f"❌ No valid percentiles extracted after 'Distribution:' anchor.\n{forecast_text[:500]}")

    if verbose:
        print("\n✅ Final extracted percentiles:", percentiles)
    return percentiles

async def get_numeric_forecast(question_details: dict, write=print) -> tuple[list[float], str]:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details.get("fine_print", "")
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper = question_details["open_upper_bound"]
    open_lower = question_details["open_lower_bound"]
    unit = question_details.get("unit", "Not stated (please infer this)")
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling.get("zero_point", None)

    upper_msg = "" if open_upper else f"The outcome can not be higher than {upper_bound}."
    lower_msg = "" if open_lower else f"The outcome can not be lower than {lower_bound}."

    async def format_and_call_claude(template):
        content = template.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
            units=unit,
        )
        return content, await call_claude(content)

    hist_task = asyncio.create_task(format_and_call_claude(NUMERIC_PROMPT_historical))
    curr_task = asyncio.create_task(format_and_call_claude(NUMERIC_PROMPT_current))
    (hist_prompt, hist_out), (curr_prompt, curr_out) = await asyncio.gather(hist_task, curr_task)

    hist_ctx_task = asyncio.create_task(process_search_queries(hist_out, forecaster_id="-1"))
    curr_ctx_task = asyncio.create_task(process_search_queries(curr_out, forecaster_id="0"))
    context_hist, context_curr = await asyncio.gather(hist_ctx_task, curr_ctx_task)

    write(f"\n[Historical LLM Output]\n{hist_out}")
    write(f"\n[Historical Context]\n{context_hist}")
    write(f"\n[Current LLM Output]\n{curr_out}")
    write(f"\n[Current Context]\n{context_curr}")

    prompt1 = NUMERIC_PROMPT_1.format(
        title=title,
        today=today,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        context=context_hist,
        units=unit,
        lower_bound_message=lower_msg,
        upper_bound_message=upper_msg,
    )

    async def run_forecaster_prompt1(f_id):
        return await call_claude(prompt1) if f_id in {"1", "2", "3"} else await call_gpt(prompt1)

    ids_prompt1 = ["1", "2", "3", "4", "5"]
    step1_outputs = await asyncio.gather(*[run_forecaster_prompt1(f_id) for f_id in ids_prompt1])
    for i, output in enumerate(step1_outputs):
        write(f"\nForecaster_{i+1} Step 1 Output:\n{output}")

    context_map = {
        "1": f"Current context: {context_curr}\nOutside view prediction: {step1_outputs[0]}",
        "2": f"Current context: {context_curr}\nOutside view prediction: {step1_outputs[3]}",
        "3": f"Current context: {context_curr}\nOutside view prediction: {step1_outputs[2]}",
        "4": f"Current context: {context_curr}\nInside view prediction: {step1_outputs[1]}",
        "5": f"Current context: {context_curr}\nInside view prediction: {step1_outputs[4]}",
    }

    def format_prompt2(f_id):
        return NUMERIC_PROMPT_2.format(
            title=title,
            today=today,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            context=context_map[f_id],
            units=unit,
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
        )

    async def run_forecaster_prompt2(f_id):
        content = format_prompt2(f_id)
        return await call_claude(content) if f_id in {"1", "2", "3"} else await call_gpt(content)

    step2_outputs = await asyncio.gather(*[run_forecaster_prompt2(f_id) for f_id in ids_prompt1])

    async def run_monte_carlo(f_id, context_input, model):
        mc_prompt = NUMERIC_PROMPT_MONTE_CARLO.format(
            title=title,
            today=today,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            context=context_input,
            units=unit,
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
        )
        raw = await model(mc_prompt)
        try:
            code_result = extract_and_run_python_code(raw)
        except Exception as e:
            code_result = f"Error running code: {e}"
        return f"{raw}\n\n--- Code Output ---\nDistribution:\n{code_result}"

    mc_6 = asyncio.create_task(run_monte_carlo("6", context_map["1"], call_claude))
    mc_7 = asyncio.create_task(run_monte_carlo("7", context_map["5"], call_gpt))
    mc_outputs = await asyncio.gather(mc_6, mc_7)

    all_outputs = step2_outputs + mc_outputs
    all_cdfs = []
    final_outputs = []
    valid_count = 0

    for i, output in enumerate(all_outputs):
        try:
            percentile_dict = extract_percentiles_from_response(output)
            cdf = generate_continuous_cdf(
                percentile_dict,
                question_type,
                open_upper,
                open_lower,
                upper_bound,
                lower_bound,
                zero_point,
            )
            all_cdfs.append(cdf)
            valid_count += 1
        except Exception as e:
            write(f"❌ Error extracting CDF from Forecaster {i+1}: {e}")
            final_outputs.append(f"=== Forecaster {i+1} ===\n❌ Failed to parse output\n")

        final_outputs.append(f"=== Forecaster {i+1} ===\nOutput:\n{output}\n")

    if valid_count > 0:
        # Convert to numpy array for vector operations
        all_cdfs_np = np.array(all_cdfs)

        # Use linear pooling to ensure valid CDFs
        combined_cdf = np.mean(all_cdfs_np, axis=0).tolist()

    else:
        combined_cdf = []
        write("❌ No valid forecasts found")

    comment = (
        f"Combined CDF: `{str(combined_cdf[:5])}...`\n\n"
        + "\n\n".join(final_outputs)
    )

    return combined_cdf, comment