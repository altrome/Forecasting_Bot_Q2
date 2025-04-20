from numeric import extract_percentiles_from_response, generate_continuous_cdf
import numpy as np

# Fake outputs from five forecasters — all valid percentiles
FAKE_FORECASTS = [
    {
        1: 7100000.0, 5: 7250000.0, 10: 7400000.0, 20: 7550000.0, 40: 7750000.0,
        60: 7950000.0, 80: 8250000.0, 90: 8500000.0, 95: 8750000.0, 99: 9200000.0
    },
    {
        1: 7000000.0, 5: 7150000.0, 10: 7300000.0, 20: 7600000.0, 40: 8000000.0,
        60: 8300000.0, 80: 8800000.0, 90: 9300000.0, 95: 9700000.0, 99: 10500000.0
    },
    {
        1: 6300000.0, 5: 6500000.0, 10: 6700000.0, 20: 6900000.0, 40: 7050000.0,
        60: 7200000.0, 80: 7500000.0, 90: 7800000.0, 95: 8000000.0, 99: 8400000.0
    },
    {
        1: 7000000.0, 5: 7200000.0, 10: 7400000.0, 20: 7600000.0, 40: 7900000.0,
        60: 8200000.0, 80: 8600000.0, 90: 8900000.0, 95: 9100000.0, 99: 10000000.0
    },
    {
        1: 6000000.0, 5: 6300000.0, 10: 6500000.0, 20: 6900000.0, 40: 7300000.0,
        60: 7700000.0, 80: 8300000.0, 90: 9000000.0, 95: 9600000.0, 99: 10500000.0
    }
]

# Define common scaling parameters
open_upper_bound = False
open_lower_bound = False
upper_bound = 12000000.0
lower_bound = 4000000.0
zero_point = None  # or something like 1.0 if using ratio grid

# Try generating CDFs
all_cdfs = []
for i, pct_dict in enumerate(FAKE_FORECASTS):
    try:
        print(f"\n--- Forecaster {i+1} ---")
        cdf = generate_continuous_cdf(
            pct_dict,
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            zero_point=zero_point
        )
        print(f"✅ CDF {i+1} generated successfully. Sample: {cdf[:5]} ...")
        all_cdfs.append((cdf, 3 if i == 4 else 1))  # triple weight on 5th
    except Exception as e:
        import traceback
        print(f"❌ Error generating CDF for Forecaster {i+1}:\n{traceback.format_exc()}")

# Combine if any valid
if all_cdfs:
    numer = sum(np.array(cdf) * weight for cdf, weight in all_cdfs)
    denom = sum(weight for _, weight in all_cdfs)
    combined_cdf = (numer / denom).tolist()
    print("\n✅ Combined CDF generated. Sample:", combined_cdf[:5])
else:
    print("\n❌ No valid CDFs found.")