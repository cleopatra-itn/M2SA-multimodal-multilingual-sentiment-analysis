from src.translation.nllb_translator import MachineTranslator
from src.language_detection.papluca_xml_roberta_model import LanguageDetector
import json
from src.utils import file_utils
import json
import pandas as pd
from glob import glob

def read_language_resources():
    lang_resources = {}
    lines = file_utils.read_file_to_list(
        "machine_translation/resources/language_codes.jsonl"
    )
    for l in lines:
        data = json.loads(l)
        lang_resources[data["iso_639_1_code"]] = data
    return lang_resources


lang_codes = read_language_resources()
# "Albanian":"sq",
low_resourced_languages = {
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Chinese": "zh",
    "Croatian": "hr",
    "Danish": "da",
    "French": "fr",
    "Hungarian": "hu",
    "Japanese": "ja",
    "Latvian": "lv",
    # "Maltese":"mt",
    "Portuguese": "pt",
    "Russian": "ru",
    "Serbian": "sr",
    "Swedish": "sv",
    "Turkish": "tr",
}

high_resourced_langauges = {
    "English": "en",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Arabic": "ar",
    "Polish": "pl",
}


if __name__ == "__main__":
    with open("log.txt","a") as outputfile:
        lang_detector = LanguageDetector()
        for source_file_path in sorted(glob("final_dataset/*.json"),reverse=False): 
            # read the file
            df_source = pd.read_json(source_file_path, lines=True)
            if not df_source.empty:
                #
                tweet_ids = df_source.tweetid.tolist()
                texts = df_source.normalized_text.astype(str).tolist()

                # language detection
                lang_ids = lang_detector.detect_language_batch(texts)

                df_source = pd.DataFrame({"tweetid": tweet_ids, "texts": texts, "langid": lang_ids})
                
                for name, group in df_source.groupby("langid"):
                    source_lang = name  
                    print(name,":")
                    if source_lang in lang_codes:
                        mt_model = MachineTranslator(lang_codes[source_lang]["flores200_code"])
                        df_list = []
                        for target_lang in low_resourced_languages.values():
                            print("Loading:", lang_codes[target_lang]["flores200_code"])

                            df_result = pd.DataFrame({"tweetid":group.tweetid.tolist()})
                            
                            target_language_code = lang_codes[target_lang]["flores200_code"]
                            translations = mt_model.process_test(texts=group.texts.tolist(), target_language_id=target_language_code)
                            
                            df_result[f"lang_{source_lang}+{target_language_code}"] = translations
                            
                            print("Done.")
                            df_list.append(df_result)
                        save_path = source_file_path.split('/')[-1]
                        pd.concat(df_list,axis=1).to_csv(f"translated_dataset/{save_path}+{name}.csv",sep="\t",index=False)
                    else:
                        outputfile.write(f"{source_lang} found in file {source_file_path} but the lang doesnt exists in translation model")
                        outputfile.write("\n")
    
