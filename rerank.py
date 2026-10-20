import argparse
import json

import numpy as np
from tqdm import tqdm

from lookback import lookback_score_tf
from models import load_model
from utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "internvl"], default="qwen")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--input_file", default="./output/bon.json")
    parser.add_argument("--output_file", default="./output/scored.json")
    args = parser.parse_args()

    alpha, lambda_ = 0.25, 1.25
    if args.model == "qwen":
        alpha = 0.5
    print(f"LookBack: alpha={alpha}, lambda={lambda_}")

    model = load_model(args.model, args.model_path, scoring=True)
    with open(args.input_file) as f:
        data = json.load(f)
    results = []
    for item in tqdm(data):
        inputs = model.prepare(item["image_path"], item["query"])
        scores = []
        for response in item["responses"]:
            outputs, output_ids, vision_token_idx = model.teacher_forcing(inputs, response)
            scores.append(lookback_score_tf(outputs, output_ids, vision_token_idx, alpha, lambda_))
            del outputs
        selected_index = int(np.argmax(scores))
        item.update(scores=scores, selected_index=selected_index,
                    selected_response=item["responses"][selected_index])
        results.append(item)
        save_json(results, args.output_file)


if __name__ == "__main__":
    main()
