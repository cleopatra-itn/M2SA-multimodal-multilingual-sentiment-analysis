from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch

# Model from https://huggingface.co/papluca/xlm-roberta-base-language-detection
class LanguageDetector:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        self.pipeline = pipeline("text-classification", model=model_ckpt, device=self.device)



    def detect_language(self, input_text: str):
        pred = self.pipeline(input_text, truncation=True, max_length=128)
        return pred


