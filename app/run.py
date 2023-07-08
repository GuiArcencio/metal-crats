from app import build_argparser
from dataio import check_metadataset_ready, load_metadataset, write_metadataset
from metaknowledge import create_metadataset, FEATURES

def run_experiment():
    parser = build_argparser()
    args = parser.parse_args()

    for feature_collection in args.features:
        if check_metadataset_ready(args.problem_type, feature_collection):
            # Metadataset already created
            meta_X, meta_y = load_metadataset(feature_collection, args.problem_type)
        else:
            meta_X = create_metadataset(
                FEATURES[feature_collection],
                args.problem_type,
                use_label_features=True,
            )
            write_metadataset(meta_X, feature_collection, args.problem_type)
            meta_X, meta_y = load_metadataset(feature_collection, args.problem_type)

        print(meta_X)