DEFAULT_MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B",
}


def load_model(name, model_path=None, scoring=False):
    import torch

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name == "qwen":
        from .qwen import Qwen

        model_class = Qwen
    else:
        from .internvl import InternVL

        model_class = InternVL
    return model_class(model_path or DEFAULT_MODELS[name], dtype=dtype, scoring=scoring)
