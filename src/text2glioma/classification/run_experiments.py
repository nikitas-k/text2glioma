from argparse import ArgumentParser

from classification.experiments import run_experiment

from ..utils import load_config

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "datalist",
        type=str,
        help="Path to the datalist JSON file.",
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        metavar=["mgmt", "1p19q", "idh", "grade"],
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--exp_type",
        type=str,
        default="real",
        metavar=["real", "synthetic", "real_synthetic"],
        help="Type of experiment to run."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Ratio of synthetic to real data for combined experiments.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, runs in debug mode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resumes training from the last checkpoint.",
    )

    return parser.parse_args()

def main(args):
    config = load_config(args.config)

    if args.experiment in ["mgmt", "1p19q", "idh", "grade"]:
        if args.exp_type in ["real", "synthetic", "real_synthetic"]:
            config["datalist"] = args.datalist
            if args.exp_type == "combined":
                config["data_ratio"] = args.ratio
            run_experiment(config, args.experiment, args.exp_type, debug=args.debug, resume=args.resume)

    else:
        raise ValueError(f"Unknown experiment: {args.experiment} with type {args.exp_type}"
                         "available experiments are: mgmt, 1p19q, idh, grade"
                         "available types are: real, synthetic, real_synthetic")
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)