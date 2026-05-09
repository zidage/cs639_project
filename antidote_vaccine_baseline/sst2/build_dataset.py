import random
import json
import os
import argparse
from pathlib import Path

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()

from datasets import Dataset, DatasetDict, load_dataset


def load_sst2_dataset():
    try:
        return load_dataset("glue", "sst2")
    except Exception as exc:
        print(f"Falling back to local SST2 cache after load_dataset failed: {exc}")
        candidates = [
            Path(os.environ["HF_DATASETS_CACHE"]) if os.environ.get("HF_DATASETS_CACHE") else None,
            Path("../cache/datasets"),
            Path(os.path.expanduser("~/.cache/huggingface/datasets")),
            Path("/root/.cache/huggingface/datasets"),
        ]
        for base in candidates:
            if not base or not base.exists():
                continue
            for validation_path in base.rglob("sst2-validation.arrow"):
                train_path = validation_path.with_name("sst2-train.arrow")
                test_path = validation_path.with_name("sst2-test.arrow")
                data = {"validation": Dataset.from_file(str(validation_path))}
                if train_path.exists():
                    data["train"] = Dataset.from_file(str(train_path))
                if test_path.exists():
                    data["test"] = Dataset.from_file(str(test_path))
                return DatasetDict(data)
        raise


dataset = load_sst2_dataset()
output_json = f'../data/sst2.json'
output_data_lst = []
for data in dataset["train"]:
    print(data)
    item = {}
    item["instruction"] = "Analyze the sentiment of the input, and respond only positive or negative"
    item["input"] = data["sentence"]
    if  data["label"] == 0: 
        item["output"] = "negative"
    else:
        item["output"] = "positive"
    output_data_lst += [item]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
