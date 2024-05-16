from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from src.utils import file_utils


# Model from https://huggingface.co/facebook/nllb-200-3.3B

class MachineTranslator:
    def __init__(self, src_lang):

        CACHE_DIR = 'hf_cache'

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", device_map="auto", src_lang=src_lang, cache_dir=CACHE_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", device_map="auto", cache_dir=CACHE_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(self, input_text:str, target_language_id):

        inputs = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        translated_tokens = self.model.generate(inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_language_id], max_length=30)
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text