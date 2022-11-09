from src.Classifier import Classifier
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the classifier")

    # Model options
    options = ["abc", "conversation"]
    parser.add_argument(
        "--model",
        type=str,
        choices=options,
        help="The model to use",
        required=True,
    )

    # How often should the classifier be run? (in frames)
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="How often should the classifier be run? (in frames)",
        min=1,
    )

    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("Interval must be greater than 0")

    return args


args = parse_args()
classifier = Classifier(model=args.model, interval=args.interval)
classifier.run()
