#!/usr/bin/env python3
import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

def parse_point(text: str):
    """Extract the first <point x=".." y=".."> tag from a string."""
    m = re.search(r'<point\s+[^>]*x="([\d\.]+)"\s+y="([\d\.]+)"', text)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

def to_device_batch(d, device):
    """Ensure all tensors have batch dimension and move to device."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            if v.dim() == 0:
                v = v.unsqueeze(0)
            elif v.dim() >= 1 and v.size(0) != 1:
                v = v.unsqueeze(0)
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

# Initialize model and processor
model_id = "allenai/Molmo-7B-D-0924"

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="auto",
)

gen_cfg = GenerationConfig(
    max_new_tokens=448,
    do_sample=False,
    stop_strings="<|endoftext|>",
)

# Inference function
def point_at_gripper(image, prompt: str = "point to the robot gripper"):
    """
    Run Molmo inference on a single image.
    
    Args:
        image_path: Path to the image file
        prompt: Instruction text (default: "point to the robot gripper")
    
    Returns:
        dict with keys 'x_pct', 'y_pct' (percentages), 'x_px', 'y_px' (pixels), 'text' (raw output)
    """
    # Load image - handle both file path and PIL Image
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        # Already a PIL Image, just ensure it's RGB
        img = image.convert("RGB")
    
    # Prepare inputs
    proc = processor.process(images=[img], text=prompt)
    proc = to_device_batch(proc, model.device)
    
    # Generate
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        out = model.generate_from_batch(
            proc, gen_cfg, tokenizer=processor.tokenizer
        )
    
    # Decode output
    gen_tokens = out[0, proc["input_ids"].size(1):]
    gen_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Parse point
    x_pct, y_pct = parse_point(gen_text)
    
    result = {"text": gen_text, "x_pct": x_pct, "y_pct": y_pct}
    
    if x_pct is not None and y_pct is not None:
        w, h = img.size
        result["x_px"] = int(round(x_pct / 100 * w))
        result["y_px"] = int(round(y_pct / 100 * h))
    else:
        result["x_px"] = None
        result["y_px"] = None
    
    return result

# Example usage:
if __name__ == "__main__":
    result = point_at_gripper("/weka/oe-training-default/jasonl/Depth-Anything-V2/test.png")
    print(f"Generated text: {result['text']}")
    print(f"Point (pixels): ({result['x_px']}, {result['y_px']})")
    print(f"Point (percent): ({result['x_pct']}, {result['y_pct']})")