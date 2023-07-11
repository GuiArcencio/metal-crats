from app import build_argparser
from dataio import check_metadataset_ready, load_metadataset, write_metadataset
from metaknowledge import create_metadataset, FEATURES

def reproduce_experiment_with_args():
    parser = build_argparser() 
    args = parser.parse_args()

    reproduce_experiment(
        args.problem_type,
        args.features,
        args.metamodels,
        args.use_label_features
    )

def reproduce_experiment(problem_type, features, metamodels, use_label_features):
    for feature_collection in features:
        if check_metadataset_ready(problem_type, feature_collection):
            # Metadataset already created
            meta_X, meta_y = load_metadataset(feature_collection, problem_type)
        else:
            meta_X = create_metadataset(
                FEATURES[feature_collection],
                problem_type,
                use_label_features=True,
            )
            write_metadataset(meta_X, feature_collection, problem_type)
            meta_X, meta_y = load_metadataset(feature_collection, problem_type)

        print(meta_X)