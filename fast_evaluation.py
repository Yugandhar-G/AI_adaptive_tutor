#!/usr/bin/env python3
"""
Fast evaluation patcher for transformer_dkt_model.py

- Creates a timestamped backup of the target file.
- Replaces the existing `evaluate_transformer_model` function with a faster batched version.
- Leaves the rest of the file untouched.

Usage:
    python fast_evaluation_fixed.py --file path/to/transformer_dkt_model.py
"""

import argparse
import datetime
import re
import sys
from pathlib import Path


def backup_file(file_path: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(file_path.suffix + f".backup_{ts}")
    backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[OK] Backup created -> {backup_path}")
    return backup_path


def replace_function(content: str) -> str:
    # Capture the entire def evaluate_transformer_model(...) block until the next top-level def or EOF
    fn_pat = re.compile(
        r'(^def\s+evaluate_transformer_model\s*\(.*?\)\s*:\s*'  # signature
        r'(?:.|\n)*?)'                                          # body (non-greedy)
        r'(?=^\s*def\s+|\Z)',                                   # up to next def or EOF
        flags=re.MULTILINE
    )

    replacement = """def evaluate_transformer_model(model, data_loader, fast_eval=False, sample_size=100, max_eval_length=200):
    \"""
    Evaluate the Transformer DKT model quickly using optional sub-sampling and truncated sequence length.
    Args:
        model: Torch model
        data_loader: Iterable of (concepts, answers, (optional features...), mask, seq_lens)
        fast_eval: If True, sub-sample batches to speed up evaluation
        sample_size: Max number of batches to evaluate when fast_eval=True
        max_eval_length: Truncate sequences to this length for speed
    Returns:
        (auc, acc) tuple. If metrics cannot be computed, returns (0.0, 0.0).
    \"""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, accuracy_score

    model.eval()
    y_true, y_prob = [], []

    disable_len = max_eval_length if isinstance(max_eval_length, int) and max_eval_length > 0 else None

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if fast_eval and i >= sample_size:
                break

            # Unpack with flexibility (some datasets may include extra feature tensors)
            concepts = batch[0]
            answers  = batch[1]
            # Optional slots (guarded)
            kw = {}
            names = ["time_features", "difficulty_features", "streak_features", "time_since_last_features", "mask", "seq_lens"]
            for idx, name in enumerate(names, start=2):
                if idx < len(batch):
                    kw[name] = batch[idx]

            # Move to model device if needed
            device = next(model.parameters()).device
            concepts = concepts.to(device)
            answers  = answers.to(device).float()
            for k in list(kw.keys()):
                if kw[k] is not None and hasattr(kw[k], "to"):
                    kw[k] = kw[k].to(device)

            # Truncate sequence length on-the-fly
            if disable_len is not None and concepts.shape[1] > disable_len:
                concepts = concepts[:, :disable_len]
                answers  = answers[:,  :disable_len]
                for k in ["time_features","difficulty_features","streak_features","time_since_last_features","mask"]:
                    if k in kw and kw[k] is not None and kw[k].shape[1] > disable_len:
                        kw[k] = kw[k][:, :disable_len]

            # Forward
            logits = model(concepts, answers, **kw)   # [B, T, num_concepts+1] or similar
            # Assume the last unit is "master" correctness prob. If not, adapt here.
            if logits.ndim == 3:
                probs = torch.sigmoid(logits[..., -1])
            else:
                probs = torch.sigmoid(logits)

            y_true.extend(answers.detach().cpu().numpy().ravel().tolist())
            y_prob.extend(probs.detach().cpu().numpy().ravel().tolist())

    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
        pred = (np.array(y_prob) >= 0.5).astype(int)
        acc = accuracy_score(y_true, pred)
        return auc, acc
    except Exception:
        return 0.0, 0.0
    """

    if not fn_pat.search(content):
        # If not found, append the function (non-destructive)
        return content.rstrip() + "\n\n" + replacement + "\n"

    content2 = fn_pat.sub(replacement, content, count=1)
    return content2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to transformer_dkt_model.py to patch")
    args = ap.parse_args()

    target = Path(args.file)
    if not target.exists():
        print(f"[ERR] File not found: {target}")
        sys.exit(1)

    backup_file(target)
    content = target.read_text(encoding="utf-8")
    updated  = replace_function(content)
    target.write_text(updated, encoding="utf-8")
    print("[OK] Patched evaluate_transformer_model() successfully.")


if __name__ == "__main__":
    main()
