import torch
from transformers import Trainer

from .base import BaseTrainer


class FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        outputs = model(**retain_inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class FT_l1(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        outputs = model(**retain_inputs)

        l1_norms = [torch.norm(p, 1) for p in model.parameters() if p.requires_grad]

        loss = outputs.loss + self.alpha * sum(l1_norms)

        return (loss, outputs) if return_outputs else loss
