from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "amaury-delille/Qwen2.5-3B-PhoMT-LoRA"
base_model_id = "Qwen/Qwen2.5-3B"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32

print(f"Loading model on device: {device}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=dtype,
    ignore_mismatched_sizes=True
)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def translate_to_vietnamese(text):
    prompt = f"Translate the following text from English to Vietnamese:\n{text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if inputs["input_ids"].shape[-1] == 0:
        raise ValueError("Tokenizer produced empty input_ids. Check tokenizer files.")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = translation.split("Translate the following text from English to Vietnamese:\n")[-1]
    translation = translation.split(text)[-1].strip()
    
    return translation

english_text = "Hello, how are you today?"
vietnamese_translation = translate_to_vietnamese(english_text)
print(f"English: {english_text}")
print(f"Vietnamese: {vietnamese_translation}")