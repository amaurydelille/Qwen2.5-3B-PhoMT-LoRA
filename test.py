from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

BASE_MODEL_ID = "Qwen/Qwen2.5-3B"
DATASET_PATH = "dataset/"
EN_SENTS_PATH = DATASET_PATH + "en_sents"
VI_SENTS_PATH = DATASET_PATH + "vi_sents"



model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

