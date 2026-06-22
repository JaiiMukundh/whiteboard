import time
import ollama

models = ["tinyllama", "smollm2", "qwen2.5:1.5b"]
prompt = "Explain quantum computing in exactly two sentences."

for model in models:
    print(f"\n==========================================")
    print(f"BENCHMARKING: {model}")
    print(f"==========================================")
    
    start_time = time.time()
    ttft = None
    mem_info = "N/A"
    response_text = ""
    
    # Stream the output to capture the first token time
    stream = ollama.generate(model=model, prompt=prompt, options={'num_predict':70}, stream=True)
    
    for chunk in stream:
        text = chunk.get('response', '') or getattr(chunk, 'response', '')
        response_text += text
        
        # When the first word is printed, capture TTFT and active memory
        if ttft is None and text.strip():
            ttft = time.time() - start_time
            try:
                # Instantly grab current model's RAM/VRAM footprint from Ollama
                for m in ollama.ps().get('models', []):
                    if model.split(':')[0] in m.get('name', ''):
                        size_gb = m.get('size', 0) / 1e9
                        mem_info = f"{size_gb:.2f} GB on {m.get('processor', 'CPU')}"
            except Exception:
                pass
        
        # When generation is finished, grab the final speed metrics
        is_done = chunk.get('done', False) or getattr(chunk, 'done', False)
        if is_done:
            load_sec = (chunk.get('load_duration', 0) or getattr(chunk, 'load_duration', 0)) / 1e9
            eval_count = chunk.get('eval_count', 0) or getattr(chunk, 'eval_count', 0)
            eval_sec = (chunk.get('eval_duration', 0) or getattr(chunk, 'eval_duration', 0)) / 1e9
            speed = eval_count / eval_sec if eval_sec > 0 else 0
            
            # Output the bare essentials
            print(f"Response:\n{response_text.strip()}\n")
            print(f"-> Startup/Load Time:    {load_sec:.4f}s")
            print(f"-> Responsiveness (TTFT): {ttft:.4f}s" if ttft else "-> Responsiveness: N/A")
            print(f"-> Memory Footprint:     {mem_info}")
            print(f"-> Inference Speed:      {speed:.2f} tokens/sec")
            
    # Unload model from memory immediately so the next model starts fresh
    ollama.generate(model=model, prompt='', keep_alive=0)
    time.sleep(2)