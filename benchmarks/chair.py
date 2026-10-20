import json
import random
from pathlib import Path


def load_data(data_path, anno_path, num_eval_samples=500):
    with open(Path(anno_path) / "instances_val2014.json") as f:
        images = json.load(f)["images"]
    images = sorted(images, key=lambda image: image["id"])
    random.shuffle(images)
    return [dict(id=image["id"], image_path=str((Path(data_path) / image["file_name"]).resolve()),
                 query="Please describe this image in detail.", responses=[])
            for image in images[:num_eval_samples]]
