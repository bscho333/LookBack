import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen:
    def __init__(self, model_path, dtype, scoring=False):
        model_kwargs = {"attn_implementation": "eager"} if scoring else {}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": "cuda:0"},
            **model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.dtype = dtype

    def prepare(self, image_path, query):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": query}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with Image.open(image_path) as image:
            return self.processor(
                text=[prompt],
                images=[image.convert("RGB")],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

    @torch.inference_mode()
    def generate(self, prepared, **generation_kwargs):
        outputs = self.model.generate(
            **prepared, **generation_kwargs, return_dict_in_generate=True,
            output_attentions=True, output_logits=True, use_cache=True,
        )
        output_ids = outputs.sequences[:, prepared["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        vision_token_idx = torch.where(
            prepared["input_ids"][0] == self.model.config.image_token_id
        )[0]
        return response, outputs, output_ids, vision_token_idx

    @torch.inference_mode()
    def teacher_forcing(self, prepared, response):
        output_ids = self.processor.tokenizer(
            response,
            return_tensors="pt",
            padding=False,
            truncation=True,
            add_special_tokens=False,
        ).input_ids.to(self.model.device)
        full_input_ids = torch.cat([prepared["input_ids"], output_ids], dim=-1)
        output_mask = torch.ones_like(output_ids, dtype=prepared["attention_mask"].dtype)
        attention_mask = torch.cat([prepared["attention_mask"], output_mask], dim=-1)
        vision_token_idx = torch.where(
            full_input_ids[0] == self.model.config.image_token_id
        )[0]
        with torch.autocast(device_type=self.model.device.type, dtype=self.dtype):
            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                pixel_values=prepared["pixel_values"].float(),
                image_grid_thw=prepared["image_grid_thw"],
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        return outputs, output_ids, vision_token_idx
