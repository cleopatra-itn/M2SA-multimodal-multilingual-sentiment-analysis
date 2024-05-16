from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

# 1. CLIP with roberta-base
# model = VisionTextDualEncoderModel.from_vision_text_pretrained(
#     "openai/clip-vit-base-patch32", "roberta-base"
# )

# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
# processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# # save the model and processor
# model.save_pretrained("clip-roberta")
# processor.save_pretrained("clip-roberta")


# 2. CLIP with xlm-roberta-base
model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "bert-base-multilingual-uncased"
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# save the model and processor
model.save_pretrained("clip-mbert")
processor.save_pretrained("clip-mbert")


