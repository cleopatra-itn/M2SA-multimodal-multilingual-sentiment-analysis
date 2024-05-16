import pandas as pd
import os


if __name__ == "__main__":
    SEEDS = [42, 123, 777, 2020, 31337]
    for language in ["ar", "bg", "bs", "da", "de", "en", "es", "fr", "hr", "hu", "it", "mt", "pl", "pt", "ru", "sr", "sv", "tr", "zh", "lv", "sq", "bg_mt", "bs_mt", "da_mt", "fr_mt", "hr_mt", "hu_mt", "mt_mt", "pt_mt", "ru_mt",  "sr_mt", "sv_mt", "tr_mt", "zh_mt"]:
        # ["lv","sq", "bg_mt", "bs_mt", "da_mt" ,"fr_mt", "hr_mt" ,"hu_mt",  "mt_mt", "pt_mt", "ru_mt" , "sr_mt" ,"sv_mt", "tr_mt", "zh_mt"]:
        # ["da", "bs" ,"bg", "tr", "sv" ,"sr" ,"pt" ,"mt" ,"hu", "fr"]:#["de","en", "es","fr","hr","hu","it","mt","pl","pt","ru","sr","sv","tr","zh"]:
        ds_collection = []
        for seed in SEEDS:
            df = pd.read_json(
                # f"./mbert-finetuned-txtnly-{language}-{seed}/all_results.json", 
                # f"./mbert-dino-finetuned-{language}-{seed}/all_results.json", 
                f"./twitter-xlmr-clip-finetuned-{language}-{seed}/all_results.json", 
                orient='index')
            ds_collection.append(df)
        print(f"language\t{language}")
        results = pd.concat(ds_collection, axis=1).T
        print(
            pd.concat([results, results.describe().loc[["mean", "std"]]]).to_csv(sep="\t"))
        print()
        print()
        print()


