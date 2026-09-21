import os
import json
import torch
from transformers import LlamaForCausalLM

model_id = "JackFram/llama-160m"
out_dir = "d:/CodeProject/L Deep-Learning/LLAMA v2/LLAMA_DK/llama-160m"
os.makedirs(out_dir, exist_ok=True)

print(f"Downloading {model_id} from Hugging Face...")
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# Extract state dict
hf_state_dict = model.state_dict()
meta_state_dict = {}

def map_key(hf_key):
    if hf_key == 'model.embed_tokens.weight':
        return 'tok_embeddings.weight'
    if hf_key == 'model.norm.weight':
        return 'norm.weight'
    if hf_key == 'lm_head.weight':
        return 'output.weight'
    
    if hf_key.startswith('model.layers.'):
        parts = hf_key.split('.')
        layer_idx = parts[2]
        sub_key = '.'.join(parts[3:])
        
        mapping = {
            'input_layernorm.weight': 'attention_norm.weight',
            'post_attention_layernorm.weight': 'ffn_norm.weight',
            'self_attn.q_proj.weight': 'attention.wq.weight',
            'self_attn.k_proj.weight': 'attention.wk.weight',
            'self_attn.v_proj.weight': 'attention.wv.weight',
            'self_attn.o_proj.weight': 'attention.wo.weight',
            'mlp.gate_proj.weight': 'feed_forward.w1.weight',
            'mlp.down_proj.weight': 'feed_forward.w2.weight',
            'mlp.up_proj.weight': 'feed_forward.w3.weight'
        }
        if sub_key in mapping:
            return f"layers.{layer_idx}.{mapping[sub_key]}"
    
    return None

print("Converting weights to Meta format...")
for k, v in hf_state_dict.items():
    new_k = map_key(k)
    if new_k is not None:
        meta_state_dict[new_k] = v
    else:
        # Ignore things like inv_freq which are computed on the fly
        if 'inv_freq' not in k:
            print(f"Warning: Unexpected key {k}")

# Save weights
out_path = os.path.join(out_dir, "consolidated.00.pth")
print(f"Saving to {out_path}...")
torch.save(meta_state_dict, out_path)

# Save params.json
params = {
    "dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "n_kv_heads": 12,
    "vocab_size": 32000,
    "multiple_of": 256,
    "norm_eps": 1e-06,
    "ffn_dim_multiplier": 1.5
}
params_path = os.path.join(out_dir, "params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=4)

print("Conversion complete!")
