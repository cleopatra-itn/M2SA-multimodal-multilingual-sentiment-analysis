from src.translation.nllb_translator import MachineTranslator
import json
from src.utils import file_utils

def read_language_resources():
    lang_resources = {}
    lines = file_utils.read_file_to_list('resources/language_codes.jsonl')
    for l in lines:
        data = json.loads(l)
        lang_resources[data['iso_639_1_code']] = data
    return lang_resources

lang_codes = read_language_resources()

# choose a specific language and load the model
lang = 'tr'

print('Loading:', lang_codes[lang]['flores200_code'])
mt_model = MachineTranslator(lang_codes[lang]['flores200_code'])
print('Done.')

# translate the source text into English
output = mt_model.process('Bana her zaman gelebilirsin.', target_language_id='eng_Latn')
print(output)