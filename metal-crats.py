import os

from app import build_argparser, reproduce_experiment

def main():
    parser = build_argparser() 
    args = parser.parse_args()

    results = reproduce_experiment(
        args.problem_type,
        args.features,
        args.metamodels,
        args.use_label_features
    )

    os.makedirs("results", exist_ok=True)
    for filename, df in results.items():
        df.to_csv(f"results/{filename}.csv")

if __name__ == "__main__":
    main()