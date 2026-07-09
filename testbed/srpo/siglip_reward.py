"""SigLIP brand-fidelity reward for SRPO (Phase 2b — gradient pressure arm).

Drop-in reward class for Tencent-Hunyuan/SRPO's trainer: exposes the same
SRP_cfg(prompt, neg_prompt, image, k) interface as their CLIP/HPS classes and
returns a per-image score in (0,1) (their ReFL threshold 0.7 assumes this
scale). Gradients flow through `image` into the generator; the judge itself
stays frozen.

Adaptation documented for the writeup: SRPO's semantic-relative pos/neg
arithmetic is text-conditioned (control words vs batch captions). This reward
is image-only — score = Platt(cos(SigLIP-tuned(image), rhode train centroid)) —
so the text arguments and k are accepted and ignored, and the trainer is run
forward-only (inversion arm disabled). That reduces SRPO to its Direct-Align
core with plain reward maximization: exactly the "direct gradient pressure on
the judge" condition Phase 2b is designed to measure.

Judge provenance: same weights + calibration as report-card judge
j3_siglip_tuned (v1) — the strongest brand-ID / violation-blind archetype,
i.e. the judge Phase 1 predicts is MOST exploitable by gradient pressure.
"""
import json

import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize


class SigLIPBrand(torch.nn.Module):
    def __init__(self, model_path="./data/siglip_tuned",
                 params_path="./data/j3_tuned_params.json",
                 device="cuda", dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel
        self.device = device
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(model_path).eval().to(device, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cal = json.loads(open(params_path).read())
        self.register_buffer("centroid", torch.tensor(cal["centroid"],
                                                      device=device, dtype=dtype))
        self.platt_a, self.platt_b = cal["platt_a"], cal["platt_b"]
        # SigLIP so400m-patch14-384 preprocessing; input is the trainer's
        # decoded image tensor in [0,1]
        self.v_pre = Compose([
            Resize(384, antialias=True),
            CenterCrop(384),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def SRP_cfg(self, prompt, neg_prompt, image_inputs, k):
        """Text args + k accepted for interface parity, unused (image-only)."""
        x = self.v_pre(image_inputs).to(self.dtype)
        z = self.model.get_image_features(pixel_values=x)
        z = z / z.norm(p=2, dim=-1, keepdim=True)
        sim = z @ self.centroid
        return torch.sigmoid(self.platt_a * sim + self.platt_b)
