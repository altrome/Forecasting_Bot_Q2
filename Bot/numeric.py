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
from llm_calls import call_claude, call_gpt, call_gpt_o3, call_gpt_o4_mini, extract_and_run_python_code
from search import process_search_queries

import re, itertools

BULLET_CHARS = "•▪●‣–*-"                    # extend as needed

VALID_KEYS = {1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99}

NUM_PATTERN = re.compile(
    r"^(?:percentile\s*)?(\d{1,3})\s*[:\-]\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$",
    re.IGNORECASE
)

def clean(s: str) -> str:
    return (
        s.strip()
         .lstrip(BULLET_CHARS)
         .replace(",", "")
         .replace(" ", "")
         .replace("−", "-")
         .lower()
    )

def generate_continuous_cdf(
    percentile_values: dict[float, float],
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None = None,
    *,
    min_step: float = 5.1e-5,
) -> list[float]:
    """
    Robust 201‑point CDF generator.
    Hard‑fails only on truly fatal issues; otherwise degrades gracefully.
    """
    from scipy.interpolate import PchipInterpolator

    # ---------- 0. Sanitise input dict ----------
    pv = {float(k): float(v) for k, v in percentile_values.items() if k is not None}
    # Drop NaNs or infinite
    pv = {k: v for k, v in pv.items() if np.isfinite(v)}

    if len(pv) < 2:          # need at least two distinct points
        raise ValueError("Need ≥2 percentile anchors")

    # ---------- 1. De‑duplicate & sort ----------
    # Small jitter for duplicate values (1e‑9 per tie)
    vals_seen = {}
    for k in sorted(pv):
        v = pv[k]
        if v in vals_seen:
            v += (len(vals_seen[v]) + 1) * 1e-9
        vals_seen.setdefault(v, []).append(k)
        pv[k] = v

    percentiles, values = zip(*sorted(pv.items()))
    percentiles = np.array(percentiles) / 100.0
    values      = np.array(values)

    if np.any(np.diff(values) <= 0):
        raise ValueError("Percentile values must be strictly increasing after de‑duplication")

    # ---------- 2. Attach bounds ----------
    if not open_lower_bound and lower_bound < values[0] - 1e-9:
        percentiles = np.insert(percentiles, 0, 0.0)
        values      = np.insert(values,      0, lower_bound)

    if not open_upper_bound and upper_bound > values[-1] + 1e-9:
        percentiles = np.append(percentiles, 1.0)
        values      = np.append(values,      upper_bound)

    # Ensure grid lies within convex hull
    effective_min = min(lower_bound, values[0])
    effective_max = max(upper_bound, values[-1])

    # ---------- 3. Optional log transform ----------
    use_log = np.all(values > 0)
    x_vals  = np.log(values) if use_log else values

    # ---------- 4. Fit spline (fallback to linear if PCHIP fails) ----------
    try:
        spline = PchipInterpolator(x_vals, percentiles, extrapolate=True)
    except Exception:
        # Linear as worst‑case fallback – always monotone
        spline = lambda x: np.interp(x, x_vals, percentiles)

    # ---------- 5. Build Metaculus grid ----------
    def grid_locations(rmin, rmax, z0):
        if z0 is None:
            s = lambda t: rmin + (rmax - rmin) * t
        else:
            if abs(rmin - z0) < 1e-6 or abs(rmax - z0) < 1e-6:
                raise ValueError(f"zero_point too close to bounds — may cause numerical instability: z0={z0}")
            ratio = (rmax - z0) / (rmin - z0)
            s = lambda t: rmin + (rmax - rmin) * ((ratio**t - 1) / (ratio - 1))
        return np.linspace(0, 1, 201, dtype=float).astype(float).tolist(), [s(t) for t in np.linspace(0, 1, 201)]

    _, cdf_x = grid_locations(lower_bound, upper_bound, zero_point)
    cdf_x = np.array(cdf_x)
    eval_x = np.log(cdf_x) if use_log else cdf_x
    eval_x_clamped = np.clip(eval_x, x_vals[0], x_vals[-1])
    
    # ---------- 6. Evaluate and enforce strict monotonicity ----------
    cdf_y = spline(eval_x_clamped).clip(0.0, 1.0)
    cdf_y = np.maximum.accumulate(cdf_y)

    # Ensure strict monotonicity once and for all
    for i in range(1, len(cdf_y)):
        if cdf_y[i] <= cdf_y[i - 1]:
            cdf_y[i] = min(cdf_y[i - 1] + min_step, 0.999999)

    cdf_y[0] = 0.0 if not open_lower_bound else max(cdf_y[0], 0.0)
    cdf_y[-1] = 1.0 if not open_upper_bound else min(cdf_y[-1], 1.0)

    # Final fail-safe
    if np.any(np.diff(cdf_y) <= 0):
        raise RuntimeError("Final monotonicity fix failed.")

    # Pin endpoints
    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0
    
    # Final step: enforce pin + clamp
    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0
    cdf_y = np.clip(cdf_y, 0.0, 1.0)
    # Final assert
    if np.any(np.diff(cdf_y) <= 0):
        raise RuntimeError("Strict monotonicity enforcement failed (should never happen)")

    return cdf_y.tolist()

def extract_percentiles_from_response(text, verbose=True):
    """
    Works whether `text` is a str or already a list of lines.
    Accepts 'Percentile1:6400000', thousands separators, scientific notation, bullets.
    """
    if isinstance(text, list):
        lines = list(itertools.chain.from_iterable(
            line.splitlines() if isinstance(line, str) else [str(line)]
            for line in text
        ))
    else:
        lines = text.splitlines()

    percentiles = {}
    collecting = False

    for idx, raw in enumerate(lines, 1):
        line = clean(str(raw))

        if not collecting and line.startswith("distribution:"):
            collecting = True
            if verbose:
                print(f"🚩 Found 'Distribution:' anchor at line {idx}")
            continue
        if not collecting:
            if verbose:
                print(f"⏭️ Skipping pre‑distribution line {idx}: {line}")
            continue

        if verbose:
            print(f"📄 Line {idx}: '{line}'")

        m = NUM_PATTERN.match(line)
        if not m:
            if verbose:
                print("⛔ No match")
            continue

        key, val_text = m.groups()
        try:
            p = int(key)
            val = float(val_text)
            if p in VALID_KEYS:
                percentiles[p] = val
                if verbose:
                    print(f"✅ Matched Percentile {p}: {val}")
            elif verbose:
                print(f"⚠️ Ignored key {p} (not in VALID_KEYS)")
        except Exception as e:
            if verbose:
                print(f"❌ Parse fail → {e}")

    if not percentiles:
        raise ValueError("❌ No valid percentiles extracted.")

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
    if not combined_cdf and valid_forecasts:
        combined_cdf = valid_forecasts[0]    # pick first good member

    if len(combined_cdf) != 201:
        raise RuntimeError(f"Final combined CDF has {len(combined_cdf)} values (should be 201)")

    comment = (
        f"Combined CDF: `{str(combined_cdf[:5])}...`\n\n"
        + "\n\n".join(final_outputs)
    )

    write(comment)

    return combined_cdf, comment
