import gc
import torch
from pathlib import Path
from typing import Dict

def reverse_convert_state_dict(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    reversed_dict = {}
    reversed_dict["tok_embeddings.weight"] = state_dict["transformer.wte.weight"].to(dtype)
    reversed_dict["output.weight"] = state_dict["lm_head.weight"].to(dtype)
    reversed_dict["norm.weight"] = state_dict["transformer.ln_f.scale"].to(dtype)

    for layer_idx in sorted(set([k.split(".")[2] for k in state_dict if k.startswith("transformer.h")])):
        # attention
        c_attn_weight = state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"].to(dtype)
        c_attn_len = c_attn_weight.shape[0] // 3
        reversed_dict[f"layers.{layer_idx}.attention.wq.weight"] = c_attn_weight[:c_attn_len]
        reversed_dict[f"layers.{layer_idx}.attention.wk.weight"] = c_attn_weight[c_attn_len:2*c_attn_len]
        reversed_dict[f"layers.{layer_idx}.attention.wv.weight"] = c_attn_weight[2*c_attn_len:]

        reversed_dict[f"layers.{layer_idx}.attention.wo.weight"] = state_dict[
            f"transformer.h.{layer_idx}.attn.c_proj.weight"
        ].to(dtype)
        # mlp
        reversed_dict[f"layers.{layer_idx}.feed_forward.w1.weight"] = state_dict[
            f"transformer.h.{layer_idx}.mlp.c_fc1.weight"
        ].to(dtype)
        reversed_dict[f"layers.{layer_idx}.feed_forward.w2.weight"] = state_dict[
            f"transformer.h.{layer_idx}.mlp.c_proj.weight"
        ].to(dtype)
        reversed_dict[f"layers.{layer_idx}.feed_forward.w3.weight"] = state_dict[
            f"transformer.h.{layer_idx}.mlp.c_fc2.weight"
        ].to(dtype)
        # rms norm
        reversed_dict[f"layers.{layer_idx}.attention_norm.weight"] = state_dict[f"transformer.h.{layer_idx}.rms_1.scale"].to(dtype)
        reversed_dict[f"layers.{layer_idx}.ffn_norm.weight"] = state_dict[f"transformer.h.{layer_idx}.rms_2.scale"].to(dtype)
    return reversed_dict

def reverse_meta_weights_for_nano_model(
    *,
    input_dir: Path = Path("checkpoints/merged"),
    output_dir: Path = Path("checkpoints/merged/reversed_model/"),
    model_size: str = "7B",
    dtype: str = "float32",
) -> None:
    # input_dir = input_dir / model_size
    # output_dir = output_dir / model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    # Load the converted checkpoint
    converted_checkpoint = torch.load(input_dir, map_location="cpu")

    # Reverse the conversion
    reversed_checkpoint = reverse_convert_state_dict(converted_checkpoint, dtype=dtype)

    # Save the reversed checkpoint
    torch.save(reversed_checkpoint, output_dir / "consolidated.00.pth")

    # del converted_checkpoint
    # del reversed_checkpoint
    gc.collect()

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(reverse_meta_weights_for_nano_model)
