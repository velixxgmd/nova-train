"""
nova_v4_test.py - Standalone trained-model tester for Nova AI V4.

Purpose:
  Load an already-trained Nova AI V4 best-checkpoint and let the user test
  text generation interactively in a terminal.

Constraints (enforced here):
  - NEVER trains, fine-tunes, or modifies weights/checkpoints/datasets/logs.
  - NEVER launches a training script or training entry point.
  - NEVER falls back to random weights on a bad checkpoint; reports a clear error.
  - Works from the tester's own folder, resolving the checkpoint relative to
    this file so a copy can live anywhere in the project.
"""
import os
import sys
import json
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: make the local project importable WITHOUT importing a TRAINING
# entry point. We import only the checkpoint + inference helpers we need.
# We deliberately avoid any training loop here.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = None

_search_dirs = []
_root = _THIS_DIR
while True:
    _search_dirs.append(_root)
    markers = [
        os.path.join(_root, "nova_v4_fast.py"),
        os.path.join(_root, "training_data.txt"),
        os.path.join(_root, "training_data2.txt"),
        os.path.join(_root, "training_data3.txt"),
        os.path.join(_root, "training_data4.txt"),
        os.path.join(_root, "training_data5.txt"),
    ]
    if any(os.path.exists(m) for m in markers):
        _PROJECT_ROOT = _root
        break
    parent = os.path.dirname(_root)
    if parent == _root:
        break
    _root = parent

if _PROJECT_ROOT is not None and _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import nova_v4_fast as _proj
except Exception as e:
    print("ERROR: unable to import project code (nova_v4_fast).")
    print(f"  working dir: {_THIS_DIR}")
    print(f"  python search dirs: {sys.path}")
    print(f"  reason: {e}")
    sys.exit(2)

NovaV4Transformer = _proj.NovaV4Transformer
_load_checkpoint = _proj.load_checkpoint
_generate_text = _proj.generate_text

# ---------------------------------------------------------------------------
# Checkpoints / paths resolved relative to THIS FILE.
# ---------------------------------------------------------------------------
_CHECKPOINT_BASENAMES = [
    "nova_v4_best_model.json",
]

def _resolve_best_checkpoint_path():
    this_dir = _THIS_DIR
    candidates = []

    for name in _CHECKPOINT_BASENAMES:
        p = os.path.join(this_dir, name)
        if os.path.exists(p):
            candidates.append(p)

    try:
        for root, dirs, files in os.walk(this_dir):
            for name in _CHECKPOINT_BASENAMES:
                if name in files:
                    p = os.path.join(root, name)
                    candidates.append(p)
            dirs[:] = [d for d in dirs if d not in ('.', '..')]
    except Exception:
        pass

    if _PROJECT_ROOT is not None:
        for name in _CHECKPOINT_BASENAMES:
            p = os.path.join(_PROJECT_ROOT, name)
            if os.path.exists(p):
                candidates.append(p)

    if not candidates:
        return None

    for name in _CHECKPOINT_BASENAMES:
        p = os.path.join(this_dir, name)
        if os.path.exists(p):
            return p
    for p in candidates:
        if _THIS_DIR in p and not p.endswith(os.path.join(this_dir, name)):
            return p
    return candidates[0]

# ---------------------------------------------------------------------------
# Checkpoint load + compatibility check.
# ---------------------------------------------------------------------------
def load_trained_checkpoint(path):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found at: {path}")
    try:
        model, meta, vocab, rev_char = _load_checkpoint(path)
    except Exception as e:
        raise RuntimeError(f"checkpoint load failed at {path}: {e}")
    # New-format checkpoints keep architecture/config at the top level, and the
    # raw JSON is easier to display from than reconstructing it via meta only.
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}
    meta = dict(meta or {})
    meta.setdefault("architecture", raw.get("architecture"))
    meta.setdefault("normalization", raw.get("normalization"))
    # config lives at checkpoint top level (not inside metadata)
    config = raw.get("config", {}) or meta.get("config", {})
    meta["config"] = config
    return model, meta, vocab, rev_char

def show_checkpoint_info(path, meta, model, vocab, rev_char):
    print("=" * 64)
    print("Nova AI V4 — Standalone Model Tester")
    print("=" * 64)
    print(f"Checkpoint : {path}")
    print(f"Epoch      : {meta.get('epoch')}")
    print(f"Train loss : {meta.get('train_loss')}")
    print(f"Val loss   : {meta.get('val_loss')}")
    print(f"Val PPL    : {meta.get('val_perplexity')}")
    print(f"Architecture : {meta.get('architecture')}, {meta.get('normalization')}")
    cfg = meta.get('config', {}) or {}
    print(f"Params         : {cfg.get('total_params')}")
    print(f"Context length : {cfg.get('context_length')}")
    print(f"Layers (L)     : {cfg.get('num_layers')}")
    print(f"Heads (H)      : {cfg.get('num_heads')}")
    print(f"d_model        : {cfg.get('d_model')}")
    print(f"d_ff           : {cfg.get('d_ff')}")
    print(f"Vocab size     : {len(vocab)} distinct characters")
    print("=" * 64)
    print("NOTE: This tester does NOT train, modify weights, or touch checkpoints.")
    print("=" * 64)
    print()

# ---------------------------------------------------------------------------
# Generation helper with controlled RNG for interactive testing.
# ---------------------------------------------------------------------------
_test_rng_state = None


def _fresh_test_seed():
    global _test_rng_state
    _test_rng_state = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    return _test_rng_state


def generate_from_model(model, prompt, vocab, rev_char, max_chars, temperature, seed):
    return _generate_text(model, prompt, vocab, rev_char, max_chars=max_chars,
                          temperature=temperature, seed=seed)


# ---------------------------------------------------------------------------
# Interactive loop.
# ---------------------------------------------------------------------------
def run_interactive(model, meta, vocab, rev_char, checkpoint_path):
    max_chars = 140
    temperature = 0.8
    prompt_seed_counter = 0

    print("Type a prompt to generate text, or a command.")
    print("Commands: /temp <value>  /length <chars>  /help  /quit")
    print()

    while True:
        try:
            raw = input("You: ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        line = raw.strip()

        if line == "":
            print("Nova: (empty prompt skipped)\n")
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("Goodbye.")
                return
            elif cmd == "/help":
                print("Nova AI V4 — Standalone Model Tester")
                print()
                print("Usage:")
                print("  Type any text prompt to see Nova's continuation.")
                print("  /temp <value>   set generation temperature (e.g. /temp 0.7)")
                print("  /length <chars> set output length (e.g. /length 200)")
                print("  /help           show this help")
                print("  /quit           exit tester")
                print()
                print(f"Current settings: max_chars={max_chars}, temperature={temperature}")
                print()
                continue
            elif cmd == "/temp":
                if len(parts) < 2:
                    print("Usage: /temp <value>")
                    print(f"Current temperature: {temperature}")
                    print()
                    continue
                try:
                    temperature = float(parts[1])
                except ValueError:
                    print(f"Invalid temperature: {parts[1]}")
                    print(f"Current temperature: {temperature}")
                    print()
                    continue
                print(f"Temperature set to {temperature}\n")
                continue
            elif cmd == "/length":
                if len(parts) < 2:
                    print("Usage: /length <chars>")
                    print(f"Current max_chars: {max_chars}")
                    print()
                    continue
                try:
                    max_chars = int(parts[1])
                except ValueError:
                    print(f"Invalid length: {parts[1]}")
                    print(f"Current max_chars: {max_chars}")
                    print()
                    continue
                if max_chars <= 0:
                    print("Length must be > 0")
                    print()
                    continue
                print(f"Output length set to {max_chars} characters\n")
                continue
            else:
                print(f"Unknown command: {cmd}. Type /help.")
                print()
                continue

        prompt_seed_counter += 1
        seed = _fresh_test_seed()
        prompt_seed_counter += 1

        print("Nova: ", end="", flush=True)
        try:
            out = generate_from_model(model, line, vocab, rev_char,
                                      max_chars=max_chars, temperature=temperature,
                                      seed=seed)
            print(out)
        except Exception as e:
            print(f"(generation error: {e})")
        print()


def main():
    ckpt_path = _resolve_best_checkpoint_path()
    if ckpt_path is None:
        print("ERROR: no trained best checkpoint found.")
        print(f"  tester folder: {_THIS_DIR}")
        if _PROJECT_ROOT is not None:
            print(f"  project root   : {_PROJECT_ROOT}")
            print("  searched for: nova_v4_best_model.json in tester folder (incl. subfolders) and project root")
        print()
        print("You must have a trained best checkpoint first. Train with:")
        print("  python nova_v4_fast.py --train --epochs <N>")
        sys.exit(1)

    print(f"Loading best checkpoint: {ckpt_path}\n")
    try:
        model, meta, vocab, rev_char = load_trained_checkpoint(ckpt_path)
    except Exception as e:
        print(f"ERROR: cannot load trained checkpoint for testing.")
        print(f"  path: {ckpt_path}")
        print(f"  reason: {e}")
        print()
        print("A fresh/uninitialized model is NOT used. If you want to test before")
        print("training, you must first create a checkpoint with:")
        print("  python nova_v4_fast.py --train --epochs <N>")
        sys.exit(1)

    show_checkpoint_info(ckpt_path, meta, model, vocab, rev_char)
    run_interactive(model, meta, vocab, rev_char, ckpt_path)

    return


if __name__ == "__main__":
    main()
