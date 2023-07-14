import argparse

def build_argparser():
    parser = argparse.ArgumentParser(
        prog="metal-crats"
    )

    parser.add_argument(
        "--problem-type",
        choices=["classification", "regression"],
        required=True,
        help="whether to consider classification or regression datasets"
    )
    parser.add_argument(
        "--features",
        choices=["minimal", "efficient", "comprehensive", "catch22"],
        nargs="+",
        help="which feature collection(s) to use"
    )
    parser.add_argument(
        "--metamodels",
        choices=["1nn", "5nn", "rf", "nb", "svm", "xgb"],
        nargs="*",
        help="which model(s) to use for metalearning"
    )
    parser.add_argument(
        "--use-label-features",
        action="store_true",
        help="whether to consider label (class/target) features"
    )

    return parser