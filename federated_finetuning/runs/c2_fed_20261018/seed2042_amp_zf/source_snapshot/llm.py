"""Decoder-LLM pieces for the C2-0 probe (docs/designs/c2_0_probe_design_2026-10-18.md).

- LLMTaskModel: a causal LM (bf16, frozen) with FP32 rank-R LoRA on all seven linear layers of every decoder
  layer. There is no new head: the task is next-token prediction of a label word, scored by the tied output
  embedding. The interface matches scra.model.TaskModel, so scra.core.local_train works unchanged.
- MNLI prompt cache: the manifest's train/validation jsonl rows (same 50 genre clients as the RoBERTa runs)
  tokenized with the model's tokenizer into the flat CachedRows format; `labels` are the label-word token ids.
- Grouped blocks: adapters have different d_in + d_out, so blocks are stacked per width D and every
  scra.core aggregation runs per group.
- Exact product-space inner products of LoRA updates from R x R matrices (no d_out x d_in materialisation).
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .core import from_blocks, to_blocks
from .data import CachedRows, derived_seed, load_cache
from .model import LoRALinear

TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
LABEL_WORDS = (" yes", " maybe", " no")        # entailment, neutral, contradiction (fedft LABELS order)
PROMPT_HEAD, PROMPT_MID, PROMPT_TAIL = "Premise: ", "\nHypothesis: ", "\nDoes the premise entail the hypothesis? Answer:"


# ---------------------------------------------------------------- model
class LoRALinearFP32(LoRALinear):
    """LoRA with FP32 factors next to a low-precision frozen base: the low-rank term runs in FP32."""

    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__(base, rank, scale)
        self.lora_a.data = self.lora_a.data.float()
        self.lora_b.data = self.lora_b.data.float()

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.lora_update(inputs.float()) * self.scale
        return self.base(inputs) + update.to(inputs.dtype)


class LLMTaskModel(nn.Module):
    def __init__(self, hf: nn.Module, rank: int, scale: float):
        super().__init__()
        self.hf = hf
        self.rank = int(rank)
        hf.requires_grad_(False)
        self.adapters: list[LoRALinearFP32] = []
        for layer in hf.model.layers:
            for path in TARGETS:
                parent_name, child = path.split(".")
                parent = getattr(layer, parent_name)
                adapter = LoRALinearFP32(getattr(parent, child), rank, scale)
                setattr(parent, child, adapter)
                self.adapters.append(adapter)

    @classmethod
    def build(cls, model_cfg: dict, rank: int, scale: float, seed: int, device: torch.device) -> "LLMTaskModel":
        from transformers import AutoModelForCausalLM
        torch.manual_seed(seed)                       # LoRA A is drawn from the torch RNG in a fixed order
        dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
        hf = AutoModelForCausalLM.from_pretrained(model_cfg["name"], revision=model_cfg.get("revision", "main"),
                                                  local_files_only=model_cfg.get("local_files_only", True),
                                                  torch_dtype=dtype,
                                                  attn_implementation=model_cfg.get("attn_implementation", "sdpa"))
        return cls(hf, rank, scale).to(device)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """FP32 next-token logits at the last real position of every row ([batch, vocab])."""
        out = self.hf.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        last = batch["attention_mask"].sum(1) - 1
        hidden = out.last_hidden_state[torch.arange(last.numel(), device=last.device), last]
        return self.hf.lm_head(hidden).float()

    def task_parameters(self) -> dict[str, nn.Parameter]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    def task_state(self) -> dict[str, Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.task_parameters().items()}

    @torch.no_grad()
    def load_task_state(self, state: dict[str, Tensor]) -> None:
        params = self.task_parameters()
        if params.keys() != state.keys():
            raise ValueError("Task state keys differ from model task parameters")
        for name, param in params.items():
            param.copy_(state[name])


# ---------------------------------------------------------------- data
def label_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(w, add_special_tokens=False) for w in LABEL_WORDS]
    if any(len(x) != 1 for x in ids):
        raise ValueError(f"label words must be single tokens: {dict(zip(LABEL_WORDS, ids))}")
    return [x[0] for x in ids]


def encode_prompt(tokenizer, premise: str, hypothesis: str, max_len: int) -> list[int]:
    """Prompt tokens; the premise is cut (never the hypothesis or the question) to fit max_len."""
    head = tokenizer.encode(PROMPT_HEAD, add_special_tokens=False)
    tail = tokenizer.encode(PROMPT_MID + hypothesis + PROMPT_TAIL, add_special_tokens=False)
    body = tokenizer.encode(premise, add_special_tokens=False)
    room = max_len - len(head) - len(tail)
    if room < 1:
        raise ValueError("hypothesis and question alone exceed max_len")
    return head + body[:room] + tail


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_fingerprint(manifest_path: Path, tokenizer_dir: str, max_len: int) -> dict:
    """Everything the prompt cache depends on; a cache built from other inputs is refused."""
    tdir = Path(tokenizer_dir)
    files = json.loads(manifest_path.read_text())["files"]
    return {"manifest_sha256": sha256_file(manifest_path),
            "train_jsonl_sha256": sha256_file(manifest_path.parent / files["train"]["path"]),
            "validation_jsonl_sha256": sha256_file(manifest_path.parent / files["validation"]["path"]),
            "tokenizer_sha256": {n: sha256_file(tdir / n) for n in TOKENIZER_FILES if (tdir / n).exists()},
            "max_len": int(max_len), "label_words": list(LABEL_WORDS),
            "template": [PROMPT_HEAD, PROMPT_MID, PROMPT_TAIL]}


def build_prompt_cache(manifest_path: Path, tokenizer_dir: str, out_dir: Path, max_len: int) -> dict:
    """Tokenize the manifest's train / validation rows once; later runs reuse the files only if every input
    (manifest, jsonl rows, tokenizer files, max_len, template, label words) and the cache files still match."""
    from transformers import AutoTokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    fingerprint = cache_fingerprint(manifest_path, tokenizer_dir, max_len)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            raise ValueError(f"prompt cache {out_dir} was built from different inputs; delete it to rebuild")
        for split in ("train", "validation"):
            path = out_dir / f"{split}.pt"
            if not path.exists() or sha256_file(path) != meta["cache_sha256"][split]:
                raise ValueError(f"prompt cache file {path} is missing or changed; delete {out_dir} to rebuild")
        return meta
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    label_ids = label_token_ids(tok)
    manifest = json.loads(manifest_path.read_text())
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stats = {}
    for split in ("train", "validation"):
        rows = [json.loads(l) for l in (manifest_path.parent / manifest["files"][split]["path"]).read_text().splitlines()
                if l.strip()]
        seqs = [encode_prompt(tok, r["text"], r["text_pair"], max_len) for r in rows]
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(s) for s in seqs])
        torch.save({"tokens": torch.tensor([t for s in seqs for t in s], dtype=torch.int32),
                    "offsets": torch.from_numpy(offsets),
                    "labels": torch.tensor([label_ids[int(r["label"])] for r in rows], dtype=torch.long),
                    "classes": torch.tensor([int(r["label"]) for r in rows], dtype=torch.long),
                    "pad_id": int(pad)}, out_dir / f"{split}.pt")
        stats[split] = {"rows": len(rows), "max_len": int(max(len(s) for s in seqs)),
                        "mean_len": float(np.mean([len(s) for s in seqs]))}
    meta = {"label_words": list(LABEL_WORDS), "label_token_ids": label_ids, "max_len": max_len,
            "tokenizer": str(tokenizer_dir), "manifest": str(manifest_path), "splits": stats,
            "fingerprint": fingerprint,
            "cache_sha256": {split: sha256_file(out_dir / f"{split}.pt") for split in ("train", "validation")}}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class LLMData:
    """Same client view as scra.data.Dataset (clients, valid_by_client, train / valid caches)."""

    def __init__(self, cache_dir: Path, manifest_path: Path):
        manifest = json.loads(manifest_path.read_text())
        self.meta = json.loads((cache_dir / "meta.json").read_text())
        self.train: CachedRows = load_cache(cache_dir / "train.pt")
        self.valid: CachedRows = load_cache(cache_dir / "validation.pt")
        valid_raw = torch.load(cache_dir / "validation.pt", map_location="cpu", weights_only=True)
        self.valid_classes = valid_raw["classes"]
        self.label_ids = torch.tensor(self.meta["label_token_ids"], dtype=torch.long)
        def read(split):
            path = manifest_path.parent / manifest["files"][split]["path"]
            return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        train_rows, rows = read("train"), read("validation")
        if len(train_rows) != len(self.train) or len(rows) != len(self.valid):
            raise ValueError("prompt cache and manifest rows differ")
        if {r.get("group") for r in train_rows} & {r.get("group") for r in rows} - {None}:
            raise ValueError("training and validation share a premise group")
        train_cache = torch.load(cache_dir / "train.pt", map_location="cpu", weights_only=True)
        label_ids = self.meta["label_token_ids"]
        if any(int(train_cache["classes"][i]) != int(r["label"]) for i, r in enumerate(train_rows)) or \
                any(int(self.valid_classes[i]) != int(r["label"]) for i, r in enumerate(rows)):
            raise ValueError("prompt cache labels do not follow the manifest rows")
        if any(int(self.train.labels[i]) != label_ids[int(r["label"])] for i, r in enumerate(train_rows[:1000])):
            raise ValueError("prompt cache label tokens do not match the label words")
        self.clients: dict[int, dict] = {}
        for client in manifest["clients"]:
            cid = int(client["id"])
            for i in client["train_indices"]:
                if train_rows[i].get("client") is not None and int(train_rows[i]["client"]) != cid:
                    raise ValueError(f"train row {i} is not owned by client {cid}")
            self.clients[int(client["id"])] = {
                "role": client["role"], "samples": int(client["samples"]),
                "train_indices": list(client["train_indices"]),
                "domain": Counter(client["domains"]).most_common(1)[0][0]}
        self.valid_by_client: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            if row.get("client") is not None:
                self.valid_by_client[int(row["client"])].append(i)

    def eval_subset(self, per_client: int, seed: int) -> dict[int, list[int]]:
        out = {}
        for c, idx in sorted(self.valid_by_client.items()):
            rng = np.random.default_rng(derived_seed(seed, "c2-eval", c))
            out[c] = sorted(rng.choice(idx, size=min(per_client, len(idx)), replace=False).tolist())
        return out


@torch.no_grad()
def evaluate_llm(model: LLMTaskModel, data: LLMData, state: dict, subset: dict[int, list[int]], device,
                 batch_size: int) -> dict:
    """3-way accuracy over the label words and full-vocabulary NLL of the gold label word, per client."""
    model.load_task_state(state)
    model.eval()
    rows = sorted(i for idx in subset.values() for i in idx)
    order = sorted(rows, key=lambda i: (data.valid.length(i), i))
    label_ids = data.label_ids.to(device)
    correct, nll = {}, {}
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        batch = data.valid.batch(ids, device)
        logits = model(batch)
        nlls = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none").cpu().tolist()
        pred = logits[:, label_ids].argmax(-1).cpu()
        for i, p, v in zip(ids, pred.tolist(), nlls):
            correct[i], nll[i] = bool(p == int(data.valid_classes[i])), v
    clients = {c: [float(np.mean([correct[i] for i in idx])), float(np.mean([nll[i] for i in idx]))]
               for c, idx in subset.items()}
    return {"client_macro_accuracy": float(np.mean([v[0] for v in clients.values()])),
            "client_macro_nll": float(np.mean([v[1] for v in clients.values()])),
            "clients": {str(c): v for c, v in sorted(clients.items())}}


@torch.no_grad()
def label_logprobs(model: LLMTaskModel, data: LLMData, state: dict, probe: list[int], device) -> Tensor:
    """Log-probabilities of the three label words on a fixed probe batch, flattened (function-space view)."""
    model.load_task_state(state)
    model.eval()
    logits = model(data.valid.batch(probe, device))
    return torch.log_softmax(logits, -1)[:, data.label_ids.to(device)].reshape(-1).double().cpu()


# ---------------------------------------------------------------- grouped blocks
class Groups:
    """Adapters grouped by block width D = d_out + d_in, preserving the global adapter order j."""

    def __init__(self, names: list[str], state: dict[str, Tensor]):
        self.names = names
        by_width: dict[int, list[int]] = OrderedDict()
        for j, n in enumerate(names):
            width = state[f"{n}.lora_b"].shape[0] + state[f"{n}.lora_a"].shape[1]
            by_width.setdefault(width, []).append(j)
        self.index = by_width                                  # width -> global adapter indices
        self.group_names = {w: [names[j] for j in js] for w, js in by_width.items()}

    def to(self, state: dict) -> dict[int, Tensor]:
        return {w: to_blocks(state, ns) for w, ns in self.group_names.items()}

    def apply(self, blocks: dict[int, Tensor], template: dict) -> dict:
        out = dict(template)
        for w, ns in self.group_names.items():
            out.update({k: v for k, v in from_blocks(blocks[w], ns, template).items() if k.split(".lora_")[0] in ns})
        return out

    def sets(self, sets_c: list[list[int]], w: int) -> list[list[int]]:
        return [sets_c[j] for j in self.index[w]]


def factors(state: dict, name: str) -> tuple[Tensor, Tensor]:
    return state[f"{name}.lora_b"].double(), state[f"{name}.lora_a"].double()


def product_gram(states: list[dict], base: dict, names: list[str]) -> np.ndarray:
    """Gram matrix <dW_x, dW_y> summed over adapters, dW_x = B_x A_x - B0 A0 (exact, via R x R products).

    <B A, C D>_F = sum((B^T C) * (A D^T)); every term needs only rank x rank matrices.
    """
    k = len(states)
    gram = np.zeros((k + 1, k + 1))
    for n in names:
        mats = [factors(base, n)] + [factors(s, n) for s in states]
        for x in range(k + 1):
            bx, ax = mats[x]
            for y in range(x, k + 1):
                by, ay = mats[y]
                v = float(((bx.T @ by) * (ax @ ay.T)).sum())
                gram[x, y] += v
                if x != y:
                    gram[y, x] += v
    # <dW_x, dW_y> = <W_x, W_y> - <W_x, W_0> - <W_0, W_y> + <W_0, W_0>
    return gram[1:, 1:] - gram[1:, :1] - gram[:1, 1:] + gram[0, 0]
