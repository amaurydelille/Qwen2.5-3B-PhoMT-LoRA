Qwen/Qwen2.5-3B fine-tuned on vinai/PhoMT dataset (10000 rows only).

## Usage

### Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "amaury-delille/Qwen2.5-3B-PhoMT-LoRA"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
```

### Translation Example

```python
def translate_to_vietnamese(text):
    prompt = f"Translate the following text from English to Vietnamese:\n{text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = translation.split("Translate the following text from English to Vietnamese:\n")[-1]
    translation = translation.split(text)[-1].strip()
    
    return translation

english_text = "Hello, how are you today?"
vietnamese_translation = translate_to_vietnamese(english_text)
print(f"English: {english_text}")
print(f"Vietnamese: {vietnamese_translation}")
```

## Citations

```bibtex
@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
      title={Qwen2 Technical Report}, 
      author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
      journal={arXiv preprint arXiv:2407.10671},
      year={2024}
}

@inproceedings{PhoMT,
title     = {{PhoMT: A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation}},
author    = {Long Doan and Linh The Nguyen and Nguyen Luong Tran and Thai Hoang and Dat Quoc Nguyen},
booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
year      = {2021}
}
```