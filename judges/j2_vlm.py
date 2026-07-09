"""J2 — frozen API VLM judges (GPT-4o, Gemini): the industry deployment archetype.

Rubric: consistency with a 6-image reference board of brand creatives (framed as
"reference set consistency", not brand identification — the refusal-safe framing
that is also the cleaner science). Per-dimension subscores 1-5 + overall 1-10,
temperature 0, JSON output. Every response cached to disk keyed by
(judge, item_id) — reruns are free. Prompt committed verbatim below.

Usage:
  python judges/j2_vlm.py smoke gpt4o        # 5 items, prints results + cost
  python judges/j2_vlm.py run gpt4o          # full test set
  python judges/j2_vlm.py run gemini
"""

import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")
CACHE = REPO / "judges" / "cache"
REFS = sorted((REPO / "judges" / "ref_board").glob("ref_*.jpg"))

RUBRIC = """You are a brand-consistency evaluator for a beauty brand's creative team.
The first 6 images are APPROVED REFERENCE creatives that define the brand's visual
identity. The LAST image is a CANDIDATE to evaluate.

Score how consistent the candidate is with the reference set on each dimension,
1 (clearly inconsistent) to 5 (fully consistent):
- palette: color palette and tonal balance
- composition: framing, negative space, layout discipline
- typography: text treatment, wordmark usage, type placement (5 if no text and
  that is consistent with the references' minimal style)
- styling: subject styling, products, art direction
- mood: lighting, mood, photographic finish

Then give overall_consistency 1-10 (10 = indistinguishable from an approved
brand creative) and one short reason.

Respond with ONLY this JSON:
{"palette": n, "composition": n, "typography": n, "styling": n, "mood": n,
 "overall_consistency": n, "reason": "..."}"""


def b64(path, max_side=None):
    if max_side:
        img = Image.open(path).convert("RGB")
        if max(img.size) > max_side:
            s = max_side / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    return base64.b64encode(Path(path).read_bytes()).decode()


def cache_path(judge, item_id):
    h = hashlib.sha1(item_id.encode()).hexdigest()[:16]
    return CACHE / judge / f"{h}.json"


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    out = json.loads(text[text.index("{"):text.rindex("}") + 1] if "{" in text else text)
    if isinstance(out, list):
        out = out[0]
    return out


class BilledError(Exception):
    """Request billed but unusable — carries usage so the tracker counts it."""
    def __init__(self, cause, usage):
        super().__init__(str(cause)[:200])
        self.usage = usage


class GPT4o:
    name = "j2_gpt4o"
    model = "gpt-4o"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.ref_content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64(r)}", "detail": "high"}}
            for r in REFS]

    def judge(self, image_path):
        content = ([{"type": "text", "text": RUBRIC}] + self.ref_content +
                   [{"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64(image_path, 1024)}",
                                   "detail": "high"}}])
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=300,
            messages=[{"role": "user", "content": content}])
        out = parse_json(r.choices[0].message.content)
        out["_usage"] = {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
        return out


class GeminiJ:
    name = "j2_gemini"
    model = "gemini-2.5-pro"  # stable: published pricing, sane quotas, reproducible ID

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.ref_parts = [self._img_part(r) for r in REFS]

    def _img_part(self, path, max_side=None):
        from google.genai import types
        data = base64.b64decode(b64(path, max_side))
        return types.Part.from_bytes(data=data, mime_type="image/jpeg")

    def judge(self, image_path):
        from google.genai import types
        parts = [RUBRIC] + self.ref_parts + [self._img_part(image_path, 1024)]
        r = self.client.models.generate_content(
            model=self.model, contents=parts,
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=2000,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=128)))
        u = r.usage_metadata
        usage = {"in": u.prompt_token_count or 0,
                 "out": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
                 "cached": getattr(u, "cached_content_token_count", 0) or 0}
        try:
            out = parse_json(r.text)
        except Exception as e:
            raise BilledError(e, usage)
        out["_usage"] = usage
        return out


JUDGES = {"gpt4o": GPT4o, "gemini": GeminiJ}


def run(judge_key, limit=None):
    judge = JUDGES[judge_key]()
    # J2_INDEX / J2_CACHE_TAG let the same judges score other pools (e.g. the
    # Phase-2 BoN candidates) with an isolated cache
    judge.name += os.environ.get("J2_CACHE_TAG", "")
    index = os.environ.get("J2_INDEX", str(REPO / "eval/testset_index.jsonl"))
    (CACHE / judge.name).mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(index)]
    if limit:
        items = items[::max(1, len(items) // limit)][:limit]
    todo = [i for i in items if not cache_path(judge.name, i["item_id"]).exists()]
    print(f"{judge.name}: {len(items)} items, {len(todo)} uncached")
    usage = {"in": 0, "out": 0, "cached": 0}
    errors = 0

    def add_usage(u):
        for k in usage:
            usage[k] += u.get(k, 0)

    def one(it):
        nonlocal errors
        for attempt in range(5):
            try:
                out = judge.judge(it["path"])
                add_usage(out["_usage"])
                cache_path(judge.name, it["item_id"]).write_text(
                    json.dumps({"item_id": it["item_id"], **out}))
                return
            except Exception as e:
                # every non-429 response was BILLED — count it even though unusable
                if isinstance(e, BilledError):
                    add_usage(e.usage)
                # parse failures at temp 0 are deterministic: one retry, then give up
                last = attempt == 4 or (isinstance(e, BilledError) and attempt >= 1)
                if last:
                    errors += 1
                    cache_path(judge.name, it["item_id"]).write_text(
                        json.dumps({"item_id": it["item_id"], "error": str(e)[:200]}))
                    return
                time.sleep(20 * (attempt + 1) if "429" in str(e) else 2 * (attempt + 1))

    # $/M tokens: (in, cached-in, out). Gemini out-rate covers thinking tokens too.
    RATES = {"gpt4o": (2.5, 2.5, 10), "gemini": (1.25, 0.31, 10)}

    def cost():
        ri, rc, ro = RATES[judge_key]
        return ((usage["in"] - usage["cached"]) / 1e6 * ri
                + usage["cached"] / 1e6 * rc + usage["out"] / 1e6 * ro)

    import os as _os
    with ThreadPoolExecutor(max_workers=int(_os.environ.get("JW", "3"))) as ex:
        for n, _ in enumerate(ex.map(one, todo)):
            if (n + 1) % 100 == 0:
                print(f"{n+1}/{len(todo)} cost-so-far ${cost():.2f} errors={errors}", flush=True)
    print(f"DONE {judge.name}: {len(todo)} calls, {errors} errors, "
          f"tokens in={usage['in']} (cached {usage['cached']}) out={usage['out']}, "
          f"session cost ${cost():.2f}")


if __name__ == "__main__":
    mode, jk = sys.argv[1], sys.argv[2]
    run(jk, limit=5 if mode == "smoke" else None)
