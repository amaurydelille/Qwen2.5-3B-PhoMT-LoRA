from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import torch
import torch.nn as nn
import math
import logging
import os

logging.basicConfig(level=logging.INFO)

BASE_MODEL_ID = "Qwen/Qwen2.5-3B"
DATASET_PATH = "dataset/"
EN_SENTS_PATH = DATASET_PATH + "en_sents"
VI_SENTS_PATH = DATASET_PATH + "vi_sents"
TOKENIZED_DATASET_PATH = DATASET_PATH + "tokenized"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-4
R = 8
ALPHA = 16

class Utils:
    @staticmethod
    def load_dataset() -> Dataset:
        with open(EN_SENTS_PATH, "r") as en_file, open(VI_SENTS_PATH, "r") as vi_file:
            en_lines = en_file.readlines()
            vi_lines = vi_file.readlines()
            dataset_formatted = {"instruction": [], "answer": []}
            for en_line, vi_line in zip(en_lines, vi_lines):
                instruction = "Translate the following text from English to Vietnamese:\n" + en_line.strip()
                answer = vi_line.strip()
                dataset_formatted["instruction"].append(instruction)
                dataset_formatted["answer"].append(answer)
        return Dataset.from_dict(dataset_formatted)

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha=16, bias=True, dtype=torch.float32) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
            self.bias.requires_grad = False
        else:
            self.bias = None

        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_features, dtype=dtype) * 0.01)
            self.B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))

            self.scaling = alpha / r
        else:
            self.A = None
            self.B = None

            self.scaling = 1.0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.r > 0:
            nn.init.normal_(self.A, std=0.01)
            nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            base = base + self.bias

        if self.r > 0:
            a = x @ self.A.transpose(-1, -2)
            lora_update = a @ self.B.transpose(-1, -2)
            return base + self.scaling * lora_update

        return base

class LoRAFineTuner:
    def __init__(self, model: nn.Module, dataset: Dataset, tokenizer: AutoTokenizer, r: int = 8, alpha: int = 16, epochs: int = 10, learning_rate: float = 1e-4, batch_size: int = 16) -> None:
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.r = r
        self.alpha = alpha
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def replace_linear_with_lora(self, model: nn.Module, target_names: List[str], r: int = 8, alpha: int = 16) -> nn.Module:
        for name, module in model.named_modules():
            if any(target in name for target in target_names):
                parent = model
                *path, last = name.split(".")
                
                for p in path:
                    parent = getattr(parent, p)

                original = getattr(parent, last)

                lora_layer = LoRALinear(
                    original.in_features,
                    original.out_features,
                    r=r,
                    alpha=alpha,
                    bias=original.bias is not None,
                    dtype=original.weight.dtype
                )

                lora_layer.weight.data = original.weight.data.clone()
                if original.bias is not None:
                    lora_layer.bias.data = original.bias.data.clone()

                setattr(parent, last, lora_layer)

        return model

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        if os.path.exists(TOKENIZED_DATASET_PATH):
            logging.info(f"Loading tokenized dataset from {TOKENIZED_DATASET_PATH}")
            tokenized_dataset = Dataset.load_from_disk(TOKENIZED_DATASET_PATH)
            tokenized_dataset.set_format(type="torch")
            return tokenized_dataset
        
        logging.info("Tokenizing dataset...")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def tokenize_function(examples):
            texts = [f"{inst}\n{ans}" for inst, ans in zip(examples["instruction"], examples["answer"])]
            return self.tokenizer(texts, padding="max_length", truncation=True, max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        tokenized_dataset.set_format(type="torch")
        
        logging.info(f"Saving tokenized dataset to {TOKENIZED_DATASET_PATH}")
        tokenized_dataset.save_to_disk(TOKENIZED_DATASET_PATH)
        
        return tokenized_dataset
    
    def freeze_model(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if "A" in name or "B" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return None

    def merge_lora_weights(self, layers: List[LoRALinear]) -> None:
        for layer in layers:
            if layer.r > 0.0:
                delta_w = layer.scaling * (layer.B @ layer.A)
                layer.weight.data += delta_w
                layer.r = 0

    def train(self) -> None:
        logging.info("Replacing linear layers with LoRA layers...")
        self.model = self.replace_linear_with_lora(self.model, target_names=["q_proj", "k_proj", "v_proj", "o_proj"], r=self.r, alpha=self.alpha)
        logging.info("Freezing model...")
        self.freeze_model(self.model)
        logging.info("Moving model to device...")
        self.model.to(DEVICE)
        logging.info("Setting model to train mode...")
        self.model.train()

        logging.info("Tokenizing dataset...")
        tokenized_dataset = self.tokenize_dataset(self.dataset)
        logging.info("Creating dataloader...")
        train_dataloader = DataLoader(tokenized_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate)
        scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=100, num_training_steps=self.epochs * len(train_dataloader))
        
        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch + 1} started")
            total_loss = 0
            total_steps = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = self.model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_steps += 1
                logging.info(f"Step {total_steps}/{len(train_dataloader)} completed with loss {loss.item()}")

            with open("metrics.csv", "a") as f:
                f.write(f"{epoch + 1},{total_loss / total_steps}\n")

        logging.info(f"Total loss: {total_loss / total_steps}")

        lora_layers = [module for module in self.model.modules() if isinstance(module, LoRALinear)]
        self.merge_lora_weights(lora_layers)
        return self.model
    
    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

if __name__ == "__main__":
    logging.info("Loading dataset...")
    dataset = Utils.load_dataset()
    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    finetuner = LoRAFineTuner(
        model=model, 
        dataset=dataset,
        tokenizer=tokenizer,
        r=R,
        alpha=ALPHA,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )
    logging.info("Training...")
    finetuner.train()
    logging.info("Saving model...")
    finetuner.save_model("lora_model")
    logging.info("Model saved successfully")