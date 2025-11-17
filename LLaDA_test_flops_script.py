import os
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from calflops import calculate_flops
from dllm_cache.cache import  dLLMCacheConfig,dLLMCache
from dllm_cache.hooks import  register_cache_LLaDA
import threading
from queue import Queue
import pandas as pd
import gc
from utils import generate
from dataclasses import asdict
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FLOPs and speed.")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", 
                    help="Name of the pretrained model to load from Hugging Face.")
    parser.add_argument("--batch_size", type=int, default=1, 
                    help="Batch size for FLOPs calculation. Default is 1.")
    parser.add_argument("--prompt_interval_steps", type=int, nargs='+', default=[100], 
                    help="List of steps intervals for caching prompts.")
    parser.add_argument("--gen_interval_steps", type=int, nargs='+', default=[7], 
                    help="List of steps intervals for caching generation.")
    parser.add_argument("--transfer_ratio", type=float, default=0.25, 
                    help="Transfer ratio for FLOPs calculation in Generate Gap.")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0], 
                    help="List of GPU IDs to run the model on (e.g., 0, 1, 2).")
    parser.add_argument("--gen_length", type=int, default=256, 
                        help="Generation length")
    parser.add_argument("--steps", type=int, default=256, 
                    help="Number of times to repeat the FLOPs calculation with cache for averaging.")
    parser.add_argument("--avg_prompt_length", type=int, default=893, 
                    help="Maximum sequence length for FLOPs calculation.")
    parser.add_argument("--cache_order", type=int, default=0, 
                    help="Maximum order for caching.")
    parser.add_argument("--prompt", type=str, default="*", help="Input prompt")
    parser.add_argument("--block_length", type=int, default=8, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale for generation")
    parser.add_argument("--remasking", type=str, default="low_confidence", help="Remasking strategy")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs to average generation time")
    parser.add_argument("--is_feature_cache", action="store_true", help="Enable feature cache")
    parser.add_argument("--is_cfg_cache", action="store_true", help="Enable cfg cache")
    return parser.parse_args()

def convert_to_float(value):
    """Convert calflops string (e.g., '500 GFLOPS' or '7.68 TMACs') to float (in FLOPS or MACs)."""
    if isinstance(value, str):
        value = value.strip()
        if "GFLOPS" in value:
            return float(value.replace("GFLOPS", "")) * 1e9
        elif "TFLOPS" in value:
            return float(value.replace("TFLOPS", "")) * 1e12
        elif "MFLOPS" in value:
            return float(value.replace("MFLOPS", "")) * 1e6
        elif "FLOPS" in value:
            return float(value.replace("FLOPS", ""))
        elif "GMACs" in value:
            return float(value.replace("GMACs", "")) * 1e9
        elif "TMACs" in value:
            return float(value.replace("TMACs", "")) * 1e12
        elif "MMACs" in value:
            return float(value.replace("MMACs", "")) * 1e6
        elif "MACs" in value:
            return float(value.replace("MACs", ""))
        else:
            return float(value)
    return float(value)

def run_experiment(gpu_id, args, prompt_interval_steps, gen_interval_steps, result_lock):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Starting experiment on GPU {gpu_id} (device: {device})")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True,max_length=2048)
    
    # Use cache
    if args.is_feature_cache:
        register_cache_LLaDA(
            model,
            "model.transformer.blocks",
            test_flops=True
        )
        print(f"Cache is enabled, prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps},transfer_ratio={args.transfer_ratio}")
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=args.transfer_ratio,
                    cfg_interval_steps=args.cfg_interval_steps if args.is_cfg_cache else 1,
                )))
    else:
        prompt_interval_steps = -1
        gen_interval_steps = -1
        args.transfer_ratio = -1
        print("Cache is not enabled")
    
    cache = dLLMCache()
    cache.reset_cache(int(args.avg_prompt_length))
    flops_cached_total = 0.0
    macs_cached_total = 0.0
    print("Starting FLOPs calculation")
    
    # Repeat FLOPs calculation with cache
    for i in range(args.steps):
        flops_cached, macs_cached, _ = calculate_flops(
            model=model,
            input_shape=(1, args.avg_prompt_length + args.gen_length),
            print_detailed=False,
            print_results=False,
            transformer_tokenizer=tokenizer,
            output_precision=4
        )
        flops_cached_val = convert_to_float(flops_cached)
        macs_cached_val = convert_to_float(macs_cached)
        flops_cached_total += flops_cached_val
        macs_cached_total += macs_cached_val

    # Calculate averages
    avg_flops_cached = flops_cached_total / args.gen_length
    avg_macs_cached = macs_cached_total /args.gen_length
    total_flops_cached = flops_cached_total
    total_macs_cached = macs_cached_total 
    
    # Show results
    result = (
        f"\n| Prompt Interval Steps: {prompt_interval_steps} | Gen Interval Steps: {gen_interval_steps}|Transfer_Ratio {args.transfer_ratio}=================\n"
        f"With Cache - FLOPs/tokens: {avg_flops_cached / 1e12:.8f} TFLOPS   MACs/tokens: {avg_macs_cached / 1e12:.8f} TMACs \n"
        f"With Cache - FLOPs: {total_flops_cached / 1e12:.4f} TFLOPS   MACs: {total_macs_cached / 1e12:.4f} TMACs \n"
    )
    print(result)

    input_ids_attn_mask = tokenizer(args.prompt, max_length=2048) 
    input_ids = torch.tensor(input_ids_attn_mask["input_ids"]).to(device).repeat(args.batch_size, args.avg_prompt_length)
    attention_mask = torch.tensor(input_ids_attn_mask["attention_mask"]).to(device).repeat(args.batch_size, args.avg_prompt_length)
    # This expands the original single-token prompt to the avg_prompt_length
    total_length = input_ids.shape[1] + args.gen_length
    print(f"Average prompt length: {input_ids.shape[1]}, Generation length: {args.gen_length}, Average total length for calculation: {total_length}")
    
    if args.is_feature_cache:
        register_cache_LLaDA(
            model,
            "model.transformer.blocks",
            test_flops=True
        )
        print(f"Cache is enabled, prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps},transfer_ratio={args.transfer_ratio}")
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=args.transfer_ratio,
                    cfg_interval_steps=args.cfg_interval_steps if args.is_cfg_cache else 1,
                )))
    
    # Warm-up (to avoid initialization overhead on the first run)
    cache.reset_cache(int(args.avg_prompt_length))
    with torch.no_grad():
        _ = generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            model=model,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            cfg_scale=args.cfg_scale
        )
    
    cache.reset_cache(int(args.avg_prompt_length))
    times = []
    for i in range(args.num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_event.record()
            out = generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model=model,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                cfg_scale=args.cfg_scale
            )
            end_event.record()
        torch.cuda.synchronize()  # Synchronize CPU and GPU
        elapsed_time_ms = start_event.elapsed_time(end_event)  # Calculate the elapsed time
        times.append(elapsed_time_ms / 1000.0)  # Convert to seconds
        print(f"Run {i+1}: {elapsed_time_ms/1000.0:.3f} seconds")
    
    arg_run_time = sum(times) / len(times) / args.batch_size/args.gen_length
    total_run_time = sum(times) / len(times) / args.batch_size
    token_per_second = args.gen_length / total_run_time
    print(f"Average generation time per token  over {args.num_runs} runs: {arg_run_time:.8f} seconds")
    
    # Prepare data for CSV
    log_entry = {
        "avg_run_time_per_token": round(arg_run_time, 8),
        "avg_token_per_second":round(token_per_second,8),
        "total_run_time":total_run_time,
        "avg_flops_per_token": round(avg_flops_cached / 1e12, 8),
        "avg_macs_per_token": round(avg_macs_cached / 1e12, 8),
        "total_flops":total_flops_cached/ 1e12, 
        "total_macs":total_macs_cached/ 1e12,
        "prompt_interval_steps": prompt_interval_steps,
        "gen_interval_steps": gen_interval_steps,
        "transfer_ratio": args.transfer_ratio,
        "steps": args.steps,
        "avg_prompt_length":args.avg_prompt_length
    }

    # Ensure the ./speed_flops directory exists
    output_dir = './speed_flops'
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'flops_results.csv')

    # print("GPU")
    # with torch.profiler.profile(
    # activities=[torch.profiler.ProfilerActivity.CUDA],
    # record_shapes=True,
    # profile_memory=True
    # ) as prof:
    #     with torch.no_grad():
    #         out = generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             model=model,
    #             steps=args.steps,
    #             gen_length=args.gen_length,
    #             block_length=args.block_length,
    #             cfg_scale=args.cfg_scale
    #         )
    #     print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    #Write to CSV with thread-safe locking
    with result_lock:
        df = pd.DataFrame([log_entry])
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode='w', header=True, index=False)
    
    del model
    torch.cuda.empty_cache()  # Clear GPU memory after experiment
    gc.collect() 

def thread_worker(gpu_queue, args, prompt_interval_steps, gen_interval_steps, result_lock):
    gpu_id = gpu_queue.get()  # Get available GPU
    try:
        run_experiment(gpu_id, args, prompt_interval_steps, gen_interval_steps, result_lock)
    finally:
        gpu_queue.put(gpu_id)  # Return the GPU to the queue

def main():
    args = parse_args()

    available_gpus = torch.cuda.device_count()
    for gpu_id in args.gpu_ids:
        if gpu_id >= available_gpus:
            raise ValueError(f"GPU ID {gpu_id} is not available. Only {available_gpus} GPUs are detected.")
    gpu_queue = Queue()
    for gpu_id in args.gpu_ids:
        gpu_queue.put(gpu_id)

    result_lock = threading.Lock()
    
    for prompt_steps in args.prompt_interval_steps:
        for gen_steps in args.gen_interval_steps:
            thread = threading.Thread(
                target=thread_worker,
                args=(gpu_queue, args, prompt_steps, gen_steps, result_lock)
            )
            thread.start()

if __name__ == "__main__":
    main()
