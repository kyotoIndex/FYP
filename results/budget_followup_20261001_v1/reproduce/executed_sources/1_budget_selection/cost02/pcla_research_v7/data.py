"""Training-only SST-2 source identities and role-isolated partitions."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import unicodedata

import numpy as np

PROTOCOL = "pcla-global-budget-20260928-v7"
FIRST_BATCH_SEEDS = (401, 409, 419, 9001, 9002, 101, 109, 127, 149, 163,
                     181, 199, 223, 241, 263, 281, 307)
ROLE_SIZES = {"client_pool": 16000, "public_training": 512,
              "cohort_select": 1024, "cohort_readout": 2048, "recipe_dev": 4096}
ACCESS = {
    "algorithm": {"client_train", "public_training"},
    "finite_bank_search": {"cohort_select"},
    "cohort_report": {"cohort_readout"},
    "research_development": {"recipe_dev"},
    "local_diagnostic": {"client_local_readout"},
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def text_group(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_train(path: Path):
    """Read only the named train path. Caller must not point this at a dev file."""
    if path.name != "train.tsv" or path.parent.name != "SST-2":
        raise ValueError("v7 first-batch source must be SST-2/train.tsv")
    raw = path.read_bytes()
    rows = []
    for i, item in enumerate(csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t")):
        if set(item) != {"sentence", "label"} or item["label"] not in {"0", "1"}:
            raise ValueError(f"unexpected SST-2 source schema at row {i}")
        rows.append({"id": f"train:{i}", "label": int(item["label"]),
                     "group": hashlib.sha256(text_group(item["sentence"]).encode()).hexdigest()})
    return rows, hashlib.sha256(raw).hexdigest()


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    return groups


def _rank(key, salt):
    return hashlib.sha256(f"{PROTOCOL}|{salt}|{key}".encode()).hexdigest()


def _stratified_targets(counts, capacity):
    total = sum(counts.values())
    wanted = {y: counts[y] * capacity / total for y in (0, 1)}
    target = {y: int(wanted[y]) for y in (0, 1)}
    for _ in range(capacity - sum(target.values())):
        y = max((0, 1), key=lambda v: (wanted[v] - target[v], -v))
        target[y] += 1
    return target


def _fill_groups(groups, capacities, salt, label_targets):
    """Salted group sample with exact capacity and stratified label targets.

    Groups are visited in hash order, independent of size; non-fitting groups
    remain outside the selected sample. A role never splits a normalized text
    group. For the first-batch SST-2 source, groups have at most two rows.
    """
    bins = {name: [] for name in capacities}
    actual = {name: Counter() for name in capacities}
    ordered = sorted(groups.items(), key=lambda pair: _rank(pair[0], salt))
    for key, members in ordered:
        labels = Counter(row["label"] for row in members)
        eligible = [name for name, cap in capacities.items()
                    if len(bins[name]) + len(members) <= cap
                    and all(actual[name][y] + labels[y] <= label_targets[name][y] for y in (0, 1))]
        if not eligible:
            continue
        choice = max(eligible, key=lambda name: (
            sum((label_targets[name][y]-actual[name][y]) / max(label_targets[name][y], 1)
                * labels[y] for y in (0, 1)),
            (capacities[name]-len(bins[name])) / capacities[name],
            _rank(key, salt + name)))
        bins[choice].extend(row["id"] for row in members)
        actual[choice].update(labels)
        if all(len(bins[name]) == cap for name, cap in capacities.items()):
            break
    if any(len(bins[name]) != cap or any(actual[name][y] != label_targets[name][y] for y in (0, 1))
           for name, cap in capacities.items()):
        raise ValueError(f"source groups cannot fill stratified role capacities: {actual}")
    return bins


def _client_targets(labels, seed):
    """Constrained Dirichlet alpha=.3 counts with exact pooled labels/capacities."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([.3, .3], size=50)[:, 1]
    wanted = p * 320
    pos = np.floor(wanted).astype(int)
    total = int(labels[1])
    # Largest-remainder correction, bounded by each client's capacity.
    while int(pos.sum()) < total:
        k = max((i for i in range(50) if pos[i] < 320),
                key=lambda i: (wanted[i]-pos[i], -i))
        pos[k] += 1
    while int(pos.sum()) > total:
        k = max((i for i in range(50) if pos[i] > 0),
                key=lambda i: (pos[i]-wanted[i], -i))
        pos[k] -= 1
    return [{0: 320-int(v), 1: int(v)} for v in pos]


def _split_client_stratified(groups, target, salt):
    """Exact 64-row grouped readout using the five possible binary group types."""
    types = {(0,): [], (1,): [], (0, 0): [], (1, 1): [], (0, 1): []}
    for key, members in groups.items():
        kind = tuple(sorted(x["label"] for x in members))
        if kind not in types:
            raise ValueError("SST-2 client has group larger than two rows")
        types[kind].append((key, members))
    for entries in types.values():
        entries.sort(key=lambda pair: _rank(pair[0], salt))
    limits = {kind: len(entries) for kind, entries in types.items()}
    def choose(target_pos):
        target_neg = 64 - target_pos
        best = None
        for mixed in range(min(limits[(0, 1)], target_neg, target_pos) + 1):
            for doubles0 in range(min(limits[(0, 0)], (target_neg-mixed)//2) + 1):
                singles0 = target_neg-mixed-2*doubles0
                if singles0 > limits[(0,)]:
                    continue
                for doubles1 in range(min(limits[(1, 1)], (target_pos-mixed)//2) + 1):
                    singles1 = target_pos-mixed-2*doubles1
                    if singles1 > limits[(1,)]:
                        continue
                    counts = {(0,): singles0, (1,): singles1,
                              (0, 0): doubles0, (1, 1): doubles1, (0, 1): mixed}
                    score = sum((counts[kind] - limits[kind] * .2)**2
                                for kind in types)
                    if best is None or score < best[0]:
                        best = (score, counts)
        return None if best is None else best[1]
    desired_pos = target[1]
    chosen = None
    used_pos = None
    for deviation in range(65):
        for pos in (desired_pos-deviation, desired_pos+deviation):
            if 0 <= pos <= 64:
                chosen = choose(pos)
                if chosen is not None:
                    used_pos = pos
                    break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError("cannot create 64-row grouped readout")
    readout = [row["id"] for kind, count in chosen.items()
               for _, members in types[kind][:count] for row in members]
    training = [row["id"] for kind, count in chosen.items()
                for _, members in types[kind][count:] for row in members]
    if len(readout) != 64 or len(training) != 256:
        raise AssertionError("grouped client split capacity mismatch")
    return training, readout, used_pos


def _assign_clients(pool, by_id, seed):
    groups = _groups(by_id[x] for x in pool)
    counts = Counter(by_id[x]["label"] for x in pool)
    targets = _client_targets(counts, seed)
    clients = {str(i): [] for i in range(50)}
    actual = [{0: 0, 1: 0} for _ in range(50)]
    rng = random.Random(int(_rank(str(seed), "client-allocation")[:16], 16))
    keys = sorted(groups, key=lambda k: (-len(groups[k]), _rank(k, str(seed))))
    for key in keys:
        members = groups[key]
        need = Counter(x["label"] for x in members)
        eligible = [i for i in range(50) if len(clients[str(i)]) + len(members) <= 320]
        if not eligible:
            raise ValueError("client capacities cannot hold source group")
        # Minimize shortfall against sampled Dirichlet count targets. Exact global
        # label counts are preserved because each source row is assigned once.
        rng.shuffle(eligible)
        i = min(eligible, key=lambda j: (sum(abs(actual[j][y]+need[y]-targets[j][y])
                                            -abs(actual[j][y]-targets[j][y]) for y in (0, 1)),
                                         -sum(targets[j][y]-actual[j][y] for y in (0, 1)), j))
        clients[str(i)].extend(x["id"] for x in members)
        for y in (0, 1): actual[i][y] += need[y]
    if any(len(v) != 320 for v in clients.values()):
        raise ValueError("client allocation failed exact 320-row capacities")
    train, readout, exceptions = {}, {}, []
    for cid, ids in clients.items():
        cg = _groups(by_id[x] for x in ids)
        local_counts = Counter(by_id[x]["label"] for x in ids)
        readout_target = _stratified_targets(local_counts, 64)
        train[cid], readout[cid], used_pos = _split_client_stratified(
            cg, readout_target, f"{seed}|{cid}|local")
        if used_pos != readout_target[1]:
            exceptions.append({"seed": seed, "client_id": cid,
                               "requested_readout_positive": readout_target[1],
                               "actual_readout_positive": used_pos,
                               "reason": "whole normalized text groups cannot meet exact stratified count"})
    return train, readout, targets, exceptions


def make_manifest(train_file: Path, seeds=FIRST_BATCH_SEEDS):
    if len(seeds) != len(set(seeds)) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("unique nonnegative simulation seeds required")
    rows, source_sha = read_train(train_file)
    by_id = {row["id"]: row for row in rows}
    groups = _groups(rows)
    source_counts = Counter(row["label"] for row in rows)
    role_targets = {role: _stratified_targets(source_counts, size)
                    for role, size in ROLE_SIZES.items()}
    role_ids = _fill_groups(groups, ROLE_SIZES, "roles", role_targets)
    partitions = {}
    all_exceptions = []
    for seed in seeds:
        train, readout, targets, exceptions = _assign_clients(role_ids["client_pool"], by_id, seed)
        all_exceptions.extend(exceptions)
        def lc(ids): return dict(sorted(Counter(by_id[x]["label"] for x in ids).items()))
        partitions[str(seed)] = {
            "client_train": train, "client_local_readout": readout,
            "label_targets": targets,
            "client_train_labels": {k: lc(v) for k, v in train.items()},
            "client_local_readout_labels": {k: lc(v) for k, v in readout.items()},
        }
    observed = {role: dict(sorted(Counter(by_id[x]["label"] for x in ids).items()))
                for role, ids in role_ids.items()}
    result = {"protocol": PROTOCOL, "source": "SST-2/train.tsv", "source_sha256": source_sha,
              "source_rows": len(rows), "source_schema": ["sentence", "label"],
              "original_sentence_id_available": False,
              "group_rule": "sha256(NFKC casefold whitespace-normalized sentence)",
              "group_sampling": "SHA256(protocol,roles,group); one/two-row groups in salted order, exact label/capacity targets",
              "near_duplicate_risk": "exact normalized groups isolated; paraphrases/substring overlap unverified",
              "group_count": len(groups),
              "mixed_label_group_count": sum(len({x['label'] for x in members}) > 1 for members in groups.values()),
              "source_label_counts": dict(source_counts),
              "role_label_targets": role_targets,
              "stratification_exceptions": all_exceptions,
              "roles": role_ids, "role_label_counts": observed,
              "partitions": partitions,
              "row_labels": {r["id"]: r["label"] for r in rows},
              "row_groups": {r["id"]: r["group"] for r in rows}}
    result["split_sha256"] = digest({k: v for k, v in result.items() if k != "split_sha256"})
    return result


def role_ids(manifest, role, actor, seed=None, client_id=None):
    if role not in ACCESS.get(actor, set()):
        raise PermissionError(f"{actor} cannot read {role}")
    if role in {"client_train", "client_local_readout"}:
        if seed is None or client_id is None:
            raise ValueError("client role requires seed and client_id")
        return list(manifest["partitions"][str(seed)][role][str(client_id)])
    return list(manifest["roles"][role])


def load_role_rows(train_file: Path, manifest, role, actor, seed=None, client_id=None):
    """Resolve permitted source IDs without opening benchmark dev or test."""
    ids = role_ids(manifest, role, actor, seed, client_id)
    rows, source_sha = read_train(train_file)
    if source_sha != manifest["source_sha256"]:
        raise ValueError("training source changed since role manifest creation")
    wanted = set(ids)
    with train_file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        selected = {f"train:{i}": (row["sentence"], None, int(row["label"]))
                    for i, row in enumerate(reader) if f"train:{i}" in wanted}
    if len(selected) != len(wanted):
        raise ValueError("source IDs missing")
    return [selected[x] for x in ids]


def build_token_cache(train_file: Path, manifest, tokenizer_path: Path, output: Path,
                      max_length=64, batch_size=512):
    """Tokenize only the selected 23,680 training IDs once per source/model/length."""
    from transformers import AutoTokenizer
    if not tokenizer_path.is_dir():
        raise ValueError("tokenizer must be a preexisting local model directory")
    tokenizer_files = sorted(p for p in tokenizer_path.iterdir() if p.is_file() and
                             ("token" in p.name or p.name in {"vocab.txt", "special_tokens_map.json"}))
    if not tokenizer_files:
        raise ValueError("no tokenizer source files found")
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tokenizer_files}
    identity = {"split_sha256": manifest["split_sha256"], "source_sha256": manifest["source_sha256"],
                "tokenizer_files": source, "max_length": max_length}
    identity_sha = digest(identity)
    metadata_path = output.with_suffix(".json")
    if output.is_file() and metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old.get("identity_sha256") == identity_sha and old.get("cache_sha256") == hashlib.sha256(output.read_bytes()).hexdigest():
            return old
        raise ValueError("existing token cache identity differs; use a new output path")
    # One role ordering is sufficient: each source row occurs in exactly one role.
    ids = [x for role in manifest["roles"].values() for x in role]
    if len(ids) != len(set(ids)) or len(ids) != 23680:
        raise ValueError("invalid frozen role source identities")
    wanted = set(ids)
    selected = {}
    with train_file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for i, row in enumerate(reader):
            key = f"train:{i}"
            if key in wanted:
                selected[key] = (row["sentence"], int(row["label"]))
    if set(selected) != wanted or hashlib.sha256(train_file.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("training source does not match role manifest")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    arrays = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
    for at in range(0, len(ids), batch_size):
        batch_ids = ids[at:at+batch_size]
        encoded = tokenizer([selected[x][0] for x in batch_ids], truncation=True,
                            padding="max_length", max_length=max_length)
        for name in arrays:
            arrays[name].append(np.asarray(encoded.get(name, [[0]*max_length]*len(batch_ids)), dtype=np.int32))
    payload = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    payload["labels"] = np.asarray([selected[x][1] for x in ids], dtype=np.int64)
    payload["source_ids"] = np.asarray(ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(output)
    meta = {"identity": identity, "identity_sha256": identity_sha,
            "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "rows": len(ids), "shape": [len(ids), max_length],
            "source_ids_sha256": digest(ids)}
    metadata_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    return meta


class TokenCache:
    """Lookup tokenized rows by authorized role IDs without re-tokenizing."""
    def __init__(self, cache_path: Path, metadata_path: Path | None = None):
        metadata_path = metadata_path or cache_path.with_suffix(".json")
        self.meta = json.loads(metadata_path.read_text())
        if hashlib.sha256(cache_path.read_bytes()).hexdigest() != self.meta["cache_sha256"]:
            raise ValueError("token cache file changed")
        self.arrays = np.load(cache_path, allow_pickle=False)
        self.index = {str(x): i for i, x in enumerate(self.arrays["source_ids"])}
        if len(self.index) != self.meta["rows"]:
            raise ValueError("duplicate token cache source IDs")

    def get(self, manifest, role, actor, seed=None, client_id=None):
        if self.meta["identity"]["split_sha256"] != manifest["split_sha256"]:
            raise ValueError("token cache and role manifest differ")
        ids = role_ids(manifest, role, actor, seed, client_id)
        offsets = [self.index[x] for x in ids]
        return {name: self.arrays[name][offsets] for name in
                ("input_ids", "attention_mask", "token_type_ids", "labels")}

    def dataset(self, manifest, role, actor, seed=None, client_id=None):
        """Construct a torch Dataset view for the legacy DataLoader interface."""
        import torch
        arrays = self.get(manifest, role, actor, seed, client_id)

        class RoleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(arrays["labels"])

            def __getitem__(self, index):
                return {name: torch.as_tensor(values[index], dtype=torch.long)
                        for name, values in arrays.items()}

        dataset = RoleDataset()
        dataset.role = role
        dataset.actor = actor
        dataset.split_sha256 = manifest["split_sha256"]
        return dataset
