import argparse

from benchmarks.src.chair import CHAIR
from utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="./output/scored.json")
    parser.add_argument("--anno_path", default="./data/coco/annotations")
    parser.add_argument("--output_file", default="./output/result.json")
    args = parser.parse_args()
    evaluator = CHAIR(args.anno_path)
    result = evaluator.compute_chair(args.input_file, "id", "selected_response")
    save_json(result, args.output_file)
    print({key: round(value * 100, 2) for key, value in result["overall_metrics"].items()})


if __name__ == "__main__":
    main()
