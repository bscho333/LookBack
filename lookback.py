import torch


def aggregate(token_logprob, visual_lookback_score, alpha=0.25, lambda_=1.25):
    """Aggregate lookback-calibrated token scores with visual relevance weights."""
    calibrated_token_score = token_logprob + alpha * visual_lookback_score.clamp(min=1e-8).log()
    token_weight = visual_lookback_score.pow(lambda_)
    return ((token_weight * calibrated_token_score).sum() / token_weight.sum()).item()


@torch.inference_mode()
def lookback_score_tf(model_output, output_ids, vision_token_idx, alpha=0.25, lambda_=1.25):
    """Compute LookBack for an existing response using teacher forcing."""
    length = output_ids.shape[1]
    logits = model_output.logits[:, -(length + 1):-1, :].float()
    token_logprob = logits.log_softmax(-1).gather(-1, output_ids.unsqueeze(-1)).flatten()
    visual_mass = torch.zeros(length, device=logits.device)
    total_mass = torch.zeros_like(visual_mass)
    for attention in model_output.attentions:
        attention = attention[0, :, -(length + 1):-1, :].float()
        visual_mass += attention[:, :, vision_token_idx].sum(dim=(0, 2))
        total_mass += attention.sum(dim=(0, 2))
    visual_lookback_score = visual_mass / total_mass.clamp(min=1e-8)
    return aggregate(token_logprob, visual_lookback_score, alpha, lambda_)


@torch.inference_mode()
def lookback_score_gen(outputs, output_ids, vision_token_idx, special_ids=(), alpha=0.25, lambda_=1.25):
    """Use raw generation logits and attention, excluding special tokens."""
    token_logprob, visual_lookback_score = [], []
    for step, token_id in enumerate(output_ids[0]):
        if token_id.item() in special_ids:
            continue
        logits = outputs.logits[step][0].float()
        token_logprob.append(logits.log_softmax(-1)[token_id])
        visual_mass = logits.new_zeros(())
        total_mass = logits.new_zeros(())
        for attention in outputs.attentions[step]:
            attention = attention[0, :, -1, :].float()
            visual_mass += attention[:, vision_token_idx].sum()
            total_mass += attention.sum()
        visual_lookback_score.append(visual_mass / total_mass.clamp(min=1e-8))
    return aggregate(torch.stack(token_logprob), torch.stack(visual_lookback_score), alpha, lambda_)
