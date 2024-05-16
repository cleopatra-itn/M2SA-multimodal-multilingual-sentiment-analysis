
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict, disable_caching

disable_caching()

from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import evaluate

logger = logging.getLogger(__name__)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py#L364
class VisionTextClassificationModel(nn.Module):
    def __init__(self, vision_text_model, classification_head):
        super(VisionTextClassificationModel, self).__init__()
        self.vision_text_model = vision_text_model
        self.classification_head = classification_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        output = self.vision_text_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        combined_output = torch.cat([output.image_embeds, output.text_embeds], dim=1)
        classification_output = self.classification_head(combined_output)

        loss = None
        # if return_loss:
        #     loss = output.loss

        return (
            (output.loss, classification_output)
            if output.loss is not None
            else classification_output
        )  # (classification_output)
