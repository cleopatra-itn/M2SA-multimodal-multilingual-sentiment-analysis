import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoModel
from typing import Optional


class VisionTextDualEncoderModel(nn.Module):
    def __init__(self, num_classes):
        super(VisionTextDualEncoderModel, self).__init__()

        # Load the XLM-RoBERTa model
        self.text_encoder = AutoModel.from_pretrained("bert-base-multilingual-uncased")

        # Define your vision model (e.g., using torchvision)
        self.vision_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")

        vision_output_dim = self.vision_encoder.config.hidden_size

        # Combine the modalities
        self.fc = nn.Linear(
            self.text_encoder.config.hidden_size + vision_output_dim, num_classes
        )

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
        # Encode text inputs
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        ).pooler_output

        # Encode vision inputs
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Concatenate text and vision features
        combined_features = torch.cat(
            (text_outputs, vision_outputs.pooler_output), dim=1
        )

        # Forward through a linear layer for classification
        logits = self.fc(combined_features)

        return {"logits": logits}
