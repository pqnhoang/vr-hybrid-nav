"""Build an acquisition fixture (memory JSON + images) from a saved scene graph (spec §11 A6).

  python eval/acquisition/make_fixture.py --from-scene-graph runs/memory_vr_h3_2/memory/final \
      --image-root runs/memory_vr_h3_2/memory/images --out eval/acquisition/fixtures/vr_h3_2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-scene-graph", required=True,
                    help="checkpoint dir (graph.json) or a scene-graph JSON file")
    ap.add_argument("--image-root", required=True, help="folder with the best-view images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirmed-only", action="store_true")
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying")
    args = ap.parse_args()
    src = Path(args.from_scene_graph)
    graph = json.loads((src / "graph.json" if src.is_dir() else src).read_text())
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    objects = [o for o in graph["objects"] if o.get("confirmed", True) or not args.confirmed_only]
    for o in objects:
        name = o.get("best_view_image") or Path(o["best_view_image_path"]).name
        dst = out / "images" / name
        if not dst.exists():
            s = Path(args.image_root).resolve() / name
            dst.symlink_to(s) if args.link else shutil.copy2(s, dst)
        o["best_view_image_path"] = f"images/{name}"
        o.pop("best_view_image", None)
        o.pop("points", None)
    keep = {o["id"] for o in objects}
    rooms = [{**r, "object_ids": [i for i in r.get("object_ids", []) if i in keep]} for r in graph.get("rooms", [])]
    for r in rooms:
        r.pop("mask_cells", None)
    fixture = {"version": graph.get("version", 1), "rooms": rooms, "objects": objects,
               "aliases": graph.get("aliases", {})}
    (out / "memory.json").write_text(json.dumps(fixture, indent=1))
    print(f"{len(objects)} objects, {len(rooms)} rooms -> {out / 'memory.json'}")


if __name__ == "__main__":
    main()
