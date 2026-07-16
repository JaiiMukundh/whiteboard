import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FT_MODEL = BASE / "models" / "qwen_lora" / "merged"

def run_evaluation(model_path, model_name, tasks):
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    results = []
    for task_name, prompt in tasks.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt

        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        
        with torch.no_grad():
            generated = model.generate(
                **{k: v.to(device) for k, v in inputs.items()},
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        completion = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append((task_name, completion))
        
    # Free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

tasks = {
    "Logical Reasoning": (
        "Solve this riddle: A father and son are in a car accident. The father dies at the scene. "
        "The boy is rushed to the hospital and needs surgery. The surgeon looks at the boy and says, "
        "'I cannot operate on this boy, he is my son.' Who is the surgeon? Explain your answer briefly."
    ),
    "General Knowledge QA": (
        "Who wrote the play 'Hamlet', what year was it roughly written, and where is the play set?"
    ),
    "General Summarization": (
        "Summarize the following text in one sentence: "
        "A new study suggests that drinking green tea daily can significantly improve cognitive function "
        "in older adults. Researchers tracked 500 participants over a five-year period and found that "
        "those who regularly consumed green tea scored 15% higher on memory and attention tests compared "
        "to those who did not drink it. The high concentration of antioxidants in green tea is believed "
        "to protect brain cells from age-related damage."
    )
}

# Run base model
base_results = run_evaluation(BASE_MODEL, "Base Qwen-2.5-1.5B-Instruct", tasks)

# Run fine-tuned model
ft_results = run_evaluation(str(FT_MODEL), "Fine-Tuned Qwen-LoRA (Merged)", tasks)

print("\n" + "=" * 80)
print("DOMAIN ADAPTATION / CATASTROPHIC FORGETTING ANALYSIS")
print("=" * 80)

for i in range(len(base_results)):
    task_name = base_results[i][0]
    base_ans = base_results[i][1]
    ft_ans = ft_results[i][1]
    
    print(f"\nTask: {task_name}")
    print("-" * 80)
    print(f"Base Model Response:\n{base_ans}")
    print("-" * 80)
    print(f"Fine-Tuned Model Response:\n{ft_ans}")
    print("=" * 80)

