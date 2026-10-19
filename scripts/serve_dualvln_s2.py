"""DualVLN System-2-only HTTP server (TASK_move_subtask.md §6.1). Runs in the InternNav env, not vlnav's.

Loads InternVLA-N1 (DualVLN) with InternNav's own classes and rebuilds the S2 prompt exactly like
`InternVLAN1AsyncAgent.step_s2` (internnav/agent/internvla_n1_agent_realworld.py): resize to
resize_w x resize_h, history placeholder, "you can see <image>." conjunction, greedy decoding,
128 new tokens. System 1 and `generate_latents` are never called. Stateless: every request carries
the instruction, the history images and (for look-down) the previous output, so replies can be
cached and replayed by the client.

    env -u LD_LIBRARY_PATH ~/miniconda3/envs/internnav/bin/python scripts/serve_dualvln_s2.py \
        --internnav ~/hoangpqn/vln/InternNav \
        --aux-ckpt-dir ~/hoangpqn/vr_vln_foca/main_baseline/InternNav/checkpoints \
        --model checkpoint/vln/InternVLA-N1-DualVLN --devices 0 --port 8001

POST /s2   form: image (PNG, original resolution), history_0..history_{n-1} (PNG, already resized),
                 look_down_image (PNG, original resolution; only with look_down),
                 json = {"instruction": str, "look_down": bool, "prev_output": str | null}
           -> {"text", "input_size": [w, h], "processed_size": [w, h], "n_history", "latency_ms"}
GET /health -> {"model", "devices", "dtype", "resize": [w, h], "attn"}
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

PROMPT = ("You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go "
          "next to stay on track? Please output the next waypoint's coordinates in the image. Please output "
          "STOP when you have successfully completed the task.")
CONJUNCTION = "you can see "                 # InternNav always uses conjunctions[0]
IMAGE = "<image>"


def build_conversation(instruction: str, n_history: int, look_down: bool, prev_output: str | None) -> list[dict]:
    """Chat turns as InternNav builds them; image slots are filled in order by the caller.

    InternNav replaces "<instruction>." (with the period) by the instruction, whose R2R-style text ends
    with ".". A missing final period is added so the prompt reads the same."""
    instruction = instruction.strip()
    instruction += "" if instruction.endswith(".") else "."
    value = PROMPT.replace("<instruction>.", instruction)
    if n_history:
        value += " These are your historical observations: " + (IMAGE + "\n") * n_history + "."
    value += f" {CONJUNCTION}{IMAGE}."
    turns = [{"role": "user", "text": value}]
    if look_down:
        turns += [{"role": "assistant", "text": prev_output or ""},
                  {"role": "user", "text": f" {CONJUNCTION}{IMAGE}."}]
    return turns


def split_and_clean(text: str) -> list[str]:
    import re
    out = []
    for part in re.split(r"(<image>)", text):
        part = part if part == IMAGE else part.replace("\n", "").strip()
        if part:
            out.append(part)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--internnav", required=True, help="InternNav source root (put on sys.path)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--aux-ckpt-dir", default=None,
                    help="folder with depth_anything_v2_metric_hypersim_vits.pth (InternNav hard-codes its own)")
    ap.add_argument("--devices", default="0", help="comma list of GPU ids; several = split (device_map auto)")
    ap.add_argument("--max-memory-gb", type=float, default=22.0, help="per GPU when splitting")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--resize-w", type=int, default=384)
    ap.add_argument("--resize-h", type=int, default=384)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.internnav).expanduser().resolve()))
    try:                                    # LongCLIP does `from pkg_resources import packaging`;
        import pkg_resources  # noqa: F401  # newer setuptools dropped pkg_resources
    except ImportError:
        import types

        import packaging.version
        sys.modules["pkg_resources"] = types.SimpleNamespace(packaging=packaging)
    import torch
    from flask import Flask, jsonify, request
    from PIL import Image
    from transformers import AutoProcessor

    import internnav.model.basemodel.internvla_n1.internvla_n1_arch as arch
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    if args.aux_ckpt_dir:                  # holds depth_anything_v2_metric_hypersim_vits.pth (S1 encoder init)
        arch.MODEL_PATH_TO = str(Path(args.aux_ckpt_dir).expanduser().resolve())

    devices = [int(d) for d in args.devices.split(",") if d.strip()]
    if len(devices) == 1:
        device_map = {"": f"cuda:{devices[0]}"}
        max_memory = None
    else:
        device_map = "auto"
        max_memory = {d: f"{args.max_memory_gb}GiB" for d in devices}
    t0 = time.time()
    model = InternVLAN1ForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                   attn_implementation=args.attn, device_map=device_map,
                                                   max_memory=max_memory)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    first = next(model.parameters()).device
    print(f"loaded {args.model} on {devices} in {time.time() - t0:.0f} s; "
          f"memory {[round(torch.cuda.memory_allocated(d) / 2**30, 1) for d in devices]} GiB", flush=True)

    app = Flask(__name__)

    def png(name: str) -> Image.Image:
        return Image.open(io.BytesIO(request.files[name].read())).convert("RGB")

    @app.get("/health")
    def health():
        return jsonify({"model": str(args.model), "devices": devices, "dtype": "bfloat16",
                        "resize": [args.resize_w, args.resize_h], "attn": args.attn})

    @app.post("/s2")
    def s2():
        t_req = time.perf_counter()
        data = json.loads(request.form["json"])
        look_down = bool(data.get("look_down", False))
        history = [png(f"history_{i}") for i in range(sum(k.startswith("history_") for k in request.files))]
        current = png("image")
        input_size = list(current.size)
        current = current.resize((args.resize_w, args.resize_h))
        images = history + [current]
        if look_down:
            ld = png("look_down_image")
            images.append(ld)                                 # look-down frames are not resized
            input_size = list(ld.size)
        conversation, k = [], 0
        for turn in build_conversation(data["instruction"], len(history), look_down, data.get("prev_output")):
            if turn["role"] == "assistant":
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": turn["text"]}]})
                continue
            content = []
            for part in split_and_clean(turn["text"]):
                if part == IMAGE:
                    content.append({"type": "image", "image": images[k]})
                    k += 1
                else:
                    content.append({"type": "text", "text": part})
            conversation.append({"role": "user", "content": content})
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt").to(first)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True,
                                 return_dict_in_generate=True).sequences
        torch.cuda.empty_cache()          # give the cached blocks back: the GPU is shared with perception
        reply = processor.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        _, gh, gw = (int(v) for v in inputs.image_grid_thw[-1])
        patch = processor.image_processor.patch_size
        res = {"text": reply, "input_size": input_size, "processed_size": [gw * patch, gh * patch],
               "n_history": len(history), "latency_ms": round(1e3 * (time.perf_counter() - t_req), 1)}
        print(json.dumps({"instruction": data["instruction"], "look_down": look_down, **res}), flush=True)
        return jsonify(res)

    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
