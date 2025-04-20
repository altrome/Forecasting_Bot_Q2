import datetime
import numpy as np
import asyncio
import re
from scipy.interpolate import PchipInterpolator
import numpy as np
from prompts import (
    NUMERIC_PROMPT_historical,
    NUMERIC_PROMPT_current,
    NUMERIC_PROMPT_1,
    NUMERIC_PROMPT_2,
    NUMERIC_PROMPT_MONTE_CARLO,
)
from llm_calls import call_claude, call_gpt, call_gpt_o3, call_gpt_o4_mini, extract_and_run_python_code
from search import process_search_queries

def generate_continuous_cdf(
    percentile_values: dict[float, float],
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float | None,
    lower_bound: float | None,
    zero_point = None
) -> list[float]:
    """
    Build a smooth, strictly‑monotone 201‑point CDF using a shape‑preserving
    cubic spline (PCHIP).
    """
    # ---------- 1. Sanitise & augment inputs ---------------------------------
    pv = {float(k): float(v) for k, v in percentile_values.items()}

    # attach closed bounds if required
    if not open_lower_bound:
        if lower_bound is None:
            raise ValueError("lower_bound must be provided when open_lower_bound is False")
        if lower_bound < min(pv.values()) - 1e-12:
            pv[0.0] = float(lower_bound)
    if not open_upper_bound:
        if upper_bound is None:
            raise ValueError("upper_bound must be provided when open_upper_bound is False")
        pv[100.0] = float(upper_bound)

    # sort by percentile and check monotone values
    percentiles, values = zip(*sorted(pv.items()))
    percentiles = np.array(percentiles) / 100.0  # convert to 0‒1
    values      = np.array(values)

    if np.any(np.diff(values) < 0):
        raise ValueError("percentile values must be non‑decreasing")

    # ---------- 2. Optional log transform for skew‑positive data -------------
    use_log = np.all(values > 0)  # only apply if everything is strictly positive
    if use_log:
        x_vals = np.log(values)
    else:
        x_vals = values  # fall back to raw scale

    # ---------- 3. Fit a shape‑preserving cubic spline p(x) ------------------
    # We model CDF p(x) directly (x -> p); PCHIP enforces monotonicity
    spline = PchipInterpolator(x_vals, percentiles, extrapolate=True)

    # ---------- 4. Create the value grid (201 points) ----------
    def generate_cdf_locations(rmin, rmax, z0):
        if z0 is None:                           # linear scale
            scale = lambda t: rmin + (rmax - rmin) * t
        else:                                    # power / log‑like scale
            deriv_ratio = (rmax - z0) / (rmin - z0)
            scale = lambda t: rmin + (rmax - rmin) * ((deriv_ratio**t - 1) / (deriv_ratio - 1))
        return [scale(t) for t in np.linspace(0, 1, 201)]

    cdf_x = np.array(generate_cdf_locations(lower_bound, upper_bound, zero_point))
    eval_x = np.log(cdf_x) if use_log else cdf_x


    # ---------- 5. Evaluate, clip, and enforce *strict* monotonicity ----------
    cdf_y = spline(eval_x).clip(0.0, 1.0)

    # 5a. Non‑decreasing
    cdf_y = np.maximum.accumulate(cdf_y)

    # 5b. Enforce a *strict* step of at least 1e‑6
    # 5b‑1. Clip open upper tail *before* enforcing min_step
    if open_upper_bound:
        cdf_y[-1] = min(cdf_y[-1], 0.999999)

    # 5b‑2. Enforce the strict step
    min_step = 1e-6
    for i in range(1, len(cdf_y)):
        if cdf_y[i] - cdf_y[i - 1] < min_step:
            cdf_y[i] = cdf_y[i - 1] + min_step

    # 5c. Pin closed endpoints exactly
    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0
    else:
        # open upper: keep < 1
        cdf_y[-1] = min(cdf_y[-1], 0.999999)

    # 5d. Final sanity check
    if len(cdf_y) != 201:
        raise RuntimeError(f"Generated CDF must have 201 values, got {len(cdf_y)}")
    if np.any(np.diff(cdf_y) <= 0):
        raise RuntimeError("Generated CDF is not strictly monotone after correction")

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
        stripped = line.strip().replace(",", "")

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
                val = float(val_raw.replace(",", ""))
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

    async def format_and_call(prompt_template):
        content = prompt_template.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
            units=unit,
        )
        return content, await call_gpt_o4_mini(content)

    hist_task = asyncio.create_task(format_and_call(NUMERIC_PROMPT_historical))
    curr_task = asyncio.create_task(format_and_call(NUMERIC_PROMPT_current))
    (hist_prompt, hist_out), (curr_prompt, curr_out) = await asyncio.gather(hist_task, curr_task)

    hist_ctx_task = asyncio.create_task(process_search_queries(hist_out, forecaster_id="-1", question_details=question_details))
    curr_ctx_task = asyncio.create_task(process_search_queries(curr_out, forecaster_id="0", question_details=question_details))
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

    forecast_funcs = [
        call_claude,
        call_claude,
        call_gpt_o4_mini,
        call_gpt_o4_mini,
        call_gpt_o3
    ]
    step1_outputs = await asyncio.gather(*[func(prompt1) for func in forecast_funcs])

    for i, output in enumerate(step1_outputs):
        write(f"\nForecaster_{i+1} Step 1 Output:\n{output}")

    context_map = {
        "1": f"Current context: {context_curr}\nOutside view prediction: {step1_outputs[0]}",
        "2": f"Current context: {context_curr}\nOutside view prediction: {step1_outputs[2]}",
        "3": f"Current context: {context_curr}\nOutside view prediction: {step1_outputs[1]}",
        "4": f"Current context: {context_curr}\nInside view prediction: {step1_outputs[3]}",
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

    step2_outputs = await asyncio.gather(*[
        forecast_funcs[i](format_prompt2(str(i+1))) for i in range(5)
    ])

    all_cdfs = []
    final_outputs = []
    valid_forecasts = []

    for i, output in enumerate(step2_outputs):
        try:
            percentile_dict = extract_percentiles_from_response(output)
            cdf = generate_continuous_cdf(
                percentile_dict,
                open_upper,
                open_lower,
                upper_bound,
                lower_bound,
                zero_point,
            )
            all_cdfs.append((cdf, 2 if i == 4 else 1))  # double weight for O3 (i == 4)
            valid_forecasts.append(cdf)
        except Exception as e:
            write(f"❌ Error extracting CDF from Forecaster {i+1}: {e}")
            final_outputs.append(f"=== Forecaster {i+1} ===\n❌ Failed to parse output\n")

        final_outputs.append(f"=== Forecaster {i+1} ===\nOutput:\n{output}\n")

    if all_cdfs:
        numer = sum(np.array(cdf) * weight for cdf, weight in all_cdfs)
        denom = sum(weight for _, weight in all_cdfs)
        combined_cdf = (numer / denom).tolist()
    else:
        combined_cdf = []
        write("❌ No valid forecasts found")

    # Optional: sanity check
    if len(combined_cdf) != 201:
        raise RuntimeError(f"Final combined CDF has {len(combined_cdf)} values (should be 201)")

    comment = (
        f"Combined CDF: `{str(combined_cdf[:5])}...`\n\n"
        + "\n\n".join(final_outputs)
    )

    write(comment)

    return combined_cdf, comment
