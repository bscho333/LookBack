import argparse

import numpy as np
from tqdm import tqdm

from benchmarks.chair import load_data
from lookback import lookback_score_gen
from models import load_model
from utils import save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "internvl"], default="qwen")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_path", default="./data/coco/val2014")
    parser.add_argument("--anno_path", default="./data/coco/annotations")
    parser.add_argument("--output_file", default="./output/bon.json")
    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.25)
    args = parser.parse_args()
    set_seed(args.seed)

    model = load_model(args.model, args.model_path, scoring=True)
    data = load_data(args.data_path, args.anno_path, args.num_eval_samples)
    generation_config = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                             top_k=0, num_beams=1, repetition_penalty=1.0,
                             max_new_tokens=args.max_new_tokens)
    results = []
    for item in tqdm(data):
        inputs = model.prepare(item["image_path"], item["query"])
        responses, scores = [], []
        for _ in range(args.N):
            response, outputs, output_ids, vision_token_idx = model.generate(inputs, **generation_config)
            value = lookback_score_gen(outputs, output_ids, vision_token_idx,
                                     model.tokenizer.all_special_ids, args.alpha, args.lambda_)
            responses.append(response)
            scores.append(value)
            del outputs
        selected_index = int(np.argmax(scores))
        item.update(responses=responses, scores=scores, selected_index=selected_index,
                    selected_response=responses[selected_index])
        results.append(item)
        save_json(results, args.output_file)


if __name__ == "__main__":
    main()
