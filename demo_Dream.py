import time
from datetime import datetime
from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache.hooks import register_cache_Dream, logout_cache_Dream
from dataclasses import asdict
from transformers import AutoModel, AutoTokenizer
import torch

# Configuration parameters
prompt_interval_steps = 100
gen_interval_steps = 7
transfer_ratio = 0.25
use_cache = True
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 256
steps = 256
max_tokens = 2048

# Load model and tokenizer
model = (
    AutoModel.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    .to(device)
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(
    "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True
)

# Initialize cache
if use_cache:
    dLLMCache.new_instance(
        **asdict(
            dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    register_cache_Dream(model, "model.layers")

# Store conversation history
conversation_history = []

def format_time():
    """Return current time in formatted string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def truncate_conversation(history, max_tokens):
    """Truncate conversation history to ensure total tokens do not exceed max_tokens"""
    total_tokens = 0
    truncated_history = []
    for msg in reversed(history):
        tokens = len(tokenizer(msg["content"])["input_ids"])
        if total_tokens + tokens <= max_tokens:
            truncated_history.insert(0, msg)
            total_tokens += tokens
        else:
            break
    return truncated_history

def print_help():
    """Print available commands"""
    print("\nAvailable commands:")
    print("  <help>       : Show this help message")
    print("  <use_cache>  : Enable cache")
    print("  <no_cache>   : Disable cache")
    print("  <clear>      : Clear conversation history")
    print("  <exit>       : Exit the program")
    print()

print("*" * 66)
print(
    f"** Answer Length: {max_new_tokens}  | Sampling Steps: {steps}  | Cache Enabled: {use_cache}"
)
print("*" * 66)
print("Type '<help>' for available commands.")

while True:
    print("\n" + "=" * 70)
    user_input = input(f"Enter your question (Cache is {'enable' if use_cache else 'disable'}, Type '<help>' for available commands): ")

    if user_input.lower() == '<exit>':
        print("Conversation ended.")
        break

    if user_input == "<help>":
        print_help()
        continue

    if user_input == "<no_cache>":
        logout_cache_Dream(model, "model.layers")
        use_cache = False
        print("Cache disabled. Please continue with your question.")
        continue

    if user_input == "<use_cache>":
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                )
            )
        )
        register_cache_Dream(model, "model.layers")
        use_cache = True
        print("Cache enabled. Please continue with your question.")
        continue

    if user_input == "<clear>":
        conversation_history = []
        print("Conversation history cleared. Please continue with your question.")
        continue

    # Record user input time
    input_time = format_time()
    conversation_history.append({"role": "user", "content": user_input, "time": input_time})

    # Truncate conversation history to ensure it does not exceed max token limit
    conversation_history = truncate_conversation(conversation_history, max_tokens)

    # Apply chat template
    formatted_input = tokenizer.apply_chat_template(
        conversation_history, add_generation_prompt=True, tokenize=False
    )

    # Encode input
    input_ids = tokenizer(formatted_input)["input_ids"]
    attention_mask = tokenizer(formatted_input)["attention_mask"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).to(device).unsqueeze(0)

    # Reset cache
    feature_cache = dLLMCache()
    feature_cache.reset_cache(input_ids.shape[1])

    # Generate response
    start_time = time.time()
    generation_ids = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.2,
        top_p=0.95,
    ).sequences[:, input_ids.shape[1]:]
    end_time = time.time()

    # Decode response
    answer = tokenizer.batch_decode(generation_ids, skip_special_tokens=True)[0]
    reply_time = format_time()

    # Store assistant response
    conversation_history.append({"role": "assistant", "content": answer, "time": reply_time})

    # Print conversation
    print(f"Dream ({reply_time}): {answer}")
    print(f"Generation Time: {end_time - start_time:.2f} seconds")