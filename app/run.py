from app import build_argparser

def run_experiment():
    parser = build_argparser()
    args = parser.parse_args()