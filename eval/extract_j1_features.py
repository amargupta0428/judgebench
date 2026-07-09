"""Extract J1 rule features for val corpus images (fitting) and all test-set items."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from testbed.corruptions import common
from judges.j1_rules import features

def run(which):
    if which == "val":
        splits = json.loads(common.SPLITS.read_text())
        rows = [{"item_id": sid, "path": str(common.image_path(sid)),
                 "label": "on" if sid.split(":")[1].split("/")[0] == "rhode" else "off"}
                for sid, sp in splits["_image_index"].items() if sp == "val"]
        out = REPO / "eval/j1_features_val.jsonl"
    else:
        rows = [json.loads(l) for l in (REPO / "eval/testset_index.jsonl").open()]
        out = REPO / "eval/j1_features_testset.jsonl"
    with out.open("w") as f:
        for i, r in enumerate(rows):
            img = common.load_base_path(r["path"]) if hasattr(common, "load_base_path") else None
            if img is None:
                from PIL import Image
                img = Image.open(r["path"]).convert("RGB")
                if max(img.size) > 1024:
                    s = 1024 / max(img.size)
                    img = img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS)
            f.write(json.dumps({"item_id": r["item_id"], "label": r.get("label", ""),
                                "features": features(img)}) + "\n")
            if (i+1) % 100 == 0:
                print(f"{i+1}/{len(rows)}", flush=True)
    print("done", out)

run(sys.argv[1])
