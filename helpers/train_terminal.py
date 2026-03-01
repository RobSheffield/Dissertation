import argparse
import datetime
import os
from stages.train_model import train_yolo



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model from the command line.")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to dataset YAML.")
    parser.add_argument("--model_info", type=str, required=True, help="Path to model info JSON.")
    parser.add_argument("--model_dir", type=str, required=True, help="Output directory for model artifacts.")
    parser.add_argument("--weights", type=str, default="yolov5m.pt", help="Pretrained weights file.")
    parser.add_argument("--img_size", type=str, default="640", help="Image size.")
    parser.add_argument("--batch_size", type=str, default="16", help="Batch size.")
    parser.add_argument("--epochs", type=str, default="50", help="Number of epochs.")
    parser.add_argument("--flips", type = bool, default = False, help="Flip metamorphoses included in training?.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_start = datetime.datetime.now().isoformat()

    os.makedirs(args.model_dir, exist_ok=True)

    train_yolo(
        data_yaml=args.data_yaml,
        model_info=args.model_info,
        training_start=training_start,
        model_dir=args.model_dir,
        weights=args.weights,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        flips = args.flips
    )


if __name__ == "__main__":
    main()



