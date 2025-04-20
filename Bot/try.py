from numeric import generate_continuous_cdf

weighted_percentiles: dict[float, float] = {
    1.0: 6.83,
    5.0: 15.83,
    10.0: 22.0,
    20.0: 30.33,
    40.0: 50.83,
    60.0: 69.67,
    80.0: 106.67,
    90.0: 121.67,
    95.0: 165.83,
    99.0: 218.33,
}

print(generate_continuous_cdf(weighted_percentiles, True, False, 200, 1))
