from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

FEATURES = {
    "basic": {
        "sum_values": None,
        "mean": None,
        "median": None,
        "standard_deviation": None,
        "variance": None,
        "minimum": None,
        "maximum": None,
        "root_mean_square": None,
    },
    "efficient": EfficientFCParameters(),
    "minimal": MinimalFCParameters(),
    "catch22": None,
}