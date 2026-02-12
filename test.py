from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "amaury-delille/Qwen2.5-3B-PhoMT-LoRA"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

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