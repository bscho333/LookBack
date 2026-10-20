# InternVL image preprocessing adapted from OpenGVLab/InternVL.
# https://github.com/OpenGVLab/InternVL
#
# MIT License
# Copyright (c) 2023 OpenGVLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVL:
    def __init__(self, model_path, dtype, scoring=False):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval().to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.dtype = dtype
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def prepare(self, image_path, query):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            transform = build_transform(input_size=448)
            images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
            pixel_values = torch.stack([transform(image) for image in images])
        return {
            "pixel_values": pixel_values.to(device=self.model.device, dtype=self.dtype),
            "query": query,
        }

    @torch.inference_mode()
    def generate(self, prepared, **generation_kwargs):
        inputs = self._prompt_inputs(prepared)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.model.conv_template.sep.strip())
        outputs = self.model.generate(
            pixel_values=prepared["pixel_values"],
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True,
            output_logits=True,
            **generation_kwargs,
        )
        output_ids = outputs.sequences
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        vision_token_idx = torch.where(inputs.input_ids[0] == self.model.img_context_token_id)[0]
        return response, outputs, output_ids, vision_token_idx

    def _prompt_inputs(self, prepared):
        question = prepared["query"]
        if "<image>" not in question:
            question = "<image>\n" + question
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = "<img>" + "<IMG_CONTEXT>" * (
            self.model.num_image_token * prepared["pixel_values"].shape[0]
        ) + "</img>"
        query = query.replace("<image>", image_tokens, 1)
        return self.tokenizer(query, return_tensors="pt").to(self.model.device)

    @torch.inference_mode()
    def teacher_forcing(self, prepared, response):
        prompt_inputs = self._prompt_inputs(prepared)
        output_encoded = self.tokenizer(response, return_tensors="pt").to(self.model.device)
        output_ids = output_encoded.input_ids
        full_input_ids = torch.cat([prompt_inputs.input_ids, output_ids], dim=-1)
        attention_mask = torch.cat([
            prompt_inputs.attention_mask, output_encoded.attention_mask,
        ], dim=-1)
        vision_token_idx = torch.where(
            prompt_inputs.input_ids[0] == self.model.img_context_token_id
        )[0]
        input_embeds = self.model.language_model.get_input_embeddings()(full_input_ids)
        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
        selected = full_input_ids.reshape(-1) == self.model.img_context_token_id
        visual_features = self.model.extract_feature(prepared["pixel_values"])
        flat_embeds[selected] = visual_features.reshape(-1, hidden_size).to(flat_embeds.device)
        input_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_size)
        with torch.autocast(device_type=self.model.device.type, dtype=self.dtype):
            outputs = self.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        return outputs, output_ids, vision_token_idx
