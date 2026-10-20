#!/usr/bin/env python3
"""
hf_recommend_and_fetch.py (v1)
==============================
1. Probes the current machine's hardware & memory (via llm_hw.py).
2. Reads an optional Hugging Face token from hf_config.json (next to this
   script) -- only needed for GATED repos (Gemma, Llama, ...).
3. Searches Hugging Face for popular GGUF models, computes which quantization
   of each actually FITS this machine's budget, and lists the 5 best picks.
4. Offers to download one of them, with a single-line percentage progress
   display (resumable), into a model directory that both Bionic and
   LM Studio read: ~/.lmstudio/models/<publisher>/<repo>/<file>.gguf
   (override with "model_dir" in hf_config.json or --dir).

Fully platform independent: Python 3.8+ standard library only -- no pip
installs, no compilers. Works on macOS (Apple Silicon incl. M5), Windows 11,
and Linux. GPU vendors supported for budgeting: NVIDIA (nvidia-smi, even off
PATH), AMD Radeon (Linux sysfs/rocm-smi and Windows WMI/registry), and Apple
Silicon unified memory.

Usage:
    python hf_recommend_and_fetch.py                 # interactive
    python hf_recommend_and_fetch.py --list-only     # just show the 5 picks
    python hf_recommend_and_fetch.py --budget 5.2
    python hf_recommend_and_fetch.py --dir "D:/LLM/models"

Config (hf_config.json, auto-created on first run):
    hf_token            optional, gated-repo access (HF settings/tokens)
    model_dir           override destination ("" = ~/.lmstudio/models)
    budget_gb_override  force a budget instead of auto-detection (null = auto)
    max_results         how many models to list (default 5)
    scan_limit          how many HF search results to evaluate (default 40)
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import llm_hw
    from llm_hw import quant_of, quant_rank, verdict
except ImportError as e:
    print(f"Missing helper module: {e}\n"
          "Keep llm_hw.py in the same folder as this script.")
    sys.exit(1)

GB = 1024 ** 3
HF = "https://huggingface.co"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "hf_config.json")

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
UA = {"User-Agent": "llm-tool-suite/1.0 (platform-independent)"}


# ------------------------------------------------------------------ config
def load_config():
    cfg = {}
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
        except (ValueError, OSError) as e:
            print(f"WARNING: could not parse {CONFIG_PATH}: {e}")
            cfg = {}
    else:
        default = {
            "_comment": ("hf_token is OPTIONAL - only for gated repos "
                         "(https://huggingface.co/settings/tokens, scope: read). "
                         "model_dir '' = default ~/.lmstudio/models. "
                         "Keep this file private."),
            "hf_token": "",
            "model_dir": "",
            "budget_gb_override": None,
            "max_results": 5,
            "scan_limit": 40,
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)
            print(f"[created config: {CONFIG_PATH}]")
            print("  - No token set: public repos work immediately. Add a token")
            print("    to 'hf_token' in that file for gated repos (Gemma/Llama).")
        except OSError as e:
            print(f"NOTE: could not write {CONFIG_PATH}: {e}")
        cfg = default
    token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or ""
    return cfg, token


def resolve_model_dir(cfg):
    d = (cfg.get("model_dir") or "").strip()
    if d:
        return os.path.abspath(os.path.expanduser(d))
    # LM Studio (and Bionic, which is built on LM Studio) default location.
    return os.path.join(os.path.expanduser("~"), ".lmstudio", "models")


# ------------------------------------------------------------- HF REST API
def http_json(url, token=None, timeout=30):
    req = urllib.request.Request(url, headers=auth_headers(token))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def auth_headers(token=None):
    h = dict(UA)
    if token:
        h["authorization"] = f"Bearer {token}"
    return h


# Curated general-purpose LLM families, searched individually so that the
# picks are actual chat/instruct models (a raw 'most downloaded GGUF' query
# surfaces audio/embedding/misc repos).
FAMILY_SEARCHES = [
    "Qwen3 gguf", "Llama-3.3 gguf", "Llama-3.2 gguf", "Gemma-3 gguf",
    "Mistral gguf", "Phi-4 gguf", "gpt-oss gguf", "Qwen2.5-Coder gguf",
    "SmolLM3 gguf", "Granite gguf",
]

# Below Q3/IQ3, quantization degrades chat models too much to recommend.
MIN_QUANT_RANK = 2   # index of "Q3" in QUANT_RANK
MIN_SIZE_GB = 0.7    # floor to skip non-LLM GGUFs (audio, embeddings, ...)

# Rough GB of weights per billion parameters per quant level, used to sanity-
# check that a file's size matches what its quant label and the model's
# parameter count imply (catches MTP heads / LoRAs / draft modules masquerading
# as full-model files).
BYTES_PER_PARAM = {
    "F32": 4.3, "BF16": 2.1, "F16": 2.1, "FP16": 2.1, "Q8": 1.06,
    "Q6": 0.85, "Q5": 0.72, "Q4": 0.58, "Q3": 0.48, "Q2": 0.36,
    "IQ1": 0.30, "IQ2": 0.36, "IQ3": 0.48, "IQ4": 0.58, "IQ5": 0.72,
    "IQ6": 0.85, "IQ8": 1.06, "MXFP4": 0.55,
}


def quant_bpq(q):
    qu = (q or "").upper()
    for k in sorted(BYTES_PER_PARAM, key=len, reverse=True):
        if qu.startswith(k):
            return BYTES_PER_PARAM[k]
    return 0.6


def size_is_plausible(size_gb, quant, repo):
    """False when the file is far smaller than its label implies (e.g. a
    0.86 GB 'F16' file in a 27B repo - that is not the full model)."""
    m = re.search(r"(\d+(?:\.\d+)?)[bB](?![A-Za-z])", repo.split("/")[-1])
    if not m:
        return True
    params_b = float(m.group(1))
    expected_gb = params_b * 1e9 * quant_bpq(quant) / GB
    if expected_gb < 1.0:
        return True                     # too small to sanity-check meaningfully
    return size_gb >= 0.5 * expected_gb


def search_gguf_repos(token, limit=40, families=None):
    """Candidate GGUF repos: curated LLM families first, generic fallback."""
    repos = {}

    def add(entries):
        for rid, dl in entries:
            if rid and rid not in repos:
                repos[rid] = dl

    for term in (families or FAMILY_SEARCHES):
        params = urllib.parse.urlencode({"search": term, "filter": "gguf",
                                         "sort": "downloads",
                                         "direction": -1, "limit": 8})
        try:
            data = http_json(f"{HF}/api/models?{params}", token=token)
        except (urllib.error.URLError, ValueError):
            continue
        add((m.get("modelId") or m.get("id"), m.get("downloads") or 0)
            for m in data)

    if not repos:
        # Generic fallback: most-downloaded GGUF repos overall.
        params = urllib.parse.urlencode({"filter": "gguf", "sort": "downloads",
                                         "direction": -1, "limit": limit})
        try:
            data = http_json(f"{HF}/api/models?{params}", token=token)
        except urllib.error.URLError as e:
            print(f"Hugging Face unreachable: {e}")
            sys.exit(1)
        add((m.get("modelId") or m.get("id"), m.get("downloads") or 0)
            for m in data)

    return sorted(repos.items(), key=lambda t: -t[1])


def repo_gguf_groups(repo, token):
    """Group a repo's .gguf files into quant 'groups' (handles sharded files).

    Returns dict: group_name -> {"quant": str, "total_gb": float,
                                 "files": [name, ...], "parts_expected": int}
    """
    owner, _, name = repo.partition("/")
    path = "/".join(urllib.parse.quote(p, safe="") for p in (owner, name))
    try:
        info = http_json(f"{HF}/api/models/{path}?blobs=true", token=token)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None            # gated - caller may hint about tokens
        return None
    except (urllib.error.URLError, ValueError):
        return None

    groups = {}
    for s in info.get("siblings") or []:
        name = s.get("rfilename") or ""
        if not name.lower().endswith(".gguf"):
            continue
        size = s.get("size") or 0
        base = os.path.basename(name)
        shard = SHARD_RE.search(base)
        if shard:
            key = SHARD_RE.sub("", base)
            expected = int(shard.group(2))
        else:
            key = base
            expected = 1
        g = groups.setdefault(key, {"quant": quant_of(base),
                                    "total": 0, "files": [],
                                    "parts_expected": expected})
        g["total"] += size
        g["files"].append(name)
    # Drop incomplete shard sets (partially uploaded repos).
    groups = {k: v for k, v in groups.items()
              if v["parts_expected"] == 1 or len(v["files"]) == v["parts_expected"]}
    # Normalize keys (case, .gguf suffix) so a shard set and its single-file
    # equivalent collapse to one entry; prefer the single file when both exist.
    out = {}
    for key, g in groups.items():
        nk = re.sub(r"\.gguf$", "", key, flags=re.I).lower()
        cur = out.get(nk)
        if cur is None or (g["parts_expected"] == 1
                           and cur["parts_expected"] != 1):
            out[nk] = g
    return out


def best_fit_quant(groups, budget_gb):
    """Largest (highest-quality) quant group whose verdict is FITS.
    Repos whose files carry no recognizable quant label are skipped: we cannot
    reliably describe (or rank) what we would be downloading."""
    def qrank(g):
        return -1 if g["quant"] == "-" else quant_rank(g["quant"])
    fits = [(qrank(g), g) for g in groups.values()
            if g["total"] > 0 and verdict(g["total"] / GB, budget_gb) == "FITS"]
    if not fits:
        return None
    rank, best = max(fits, key=lambda t: t[0])
    return best if rank >= 0 else None


def family_key(repo):
    """Rough dedupe key so 5 different authors of Qwen3-4B don't fill the list."""
    name = repo.split("/", 1)[-1]
    name = re.sub(r"-GGUF$", "", name, flags=re.I)
    name = re.sub(r"-(Instruct|Chat|it|HF|AWQ|GPTQ)$", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def discover_models(cfg, token, budget_gb):
    limit = int(cfg.get("scan_limit") or 40)
    want = int(cfg.get("max_results") or 5)
    repos = search_gguf_repos(token, limit=limit)
    picks, seen = [], set()
    gated_hit = errors = 0
    for i, (repo, downloads) in enumerate(repos, 1):
        if len(picks) >= want:
            break
        print(f"\r  scanning Hugging Face... {i}/{len(repos)}"
              f"  ({len(picks)} fits so far)", end="", flush=True)
        groups = repo_gguf_groups(repo, token)
        if groups is None:
            gated_hit += 1
            errors += 1
            continue
        best = best_fit_quant(groups, budget_gb)
        if not best:
            continue
        # Quality gates: skip heavily-degraded quants and non-LLM tiny files.
        if quant_rank(best["quant"]) < MIN_QUANT_RANK:
            continue
        size_gb = best["total"] / GB
        if size_gb < MIN_SIZE_GB:
            continue
        if not size_is_plausible(size_gb, best["quant"], repo):
            continue
        # A full-precision chat model under 2 GB would be <1B params - that is
        # a draft/MTP module or a LoRA, not a useful pick.
        if best["quant"] in ("F16", "BF16", "FP16", "F32") and size_gb < 2.0:
            continue
        fam = family_key(repo)
        if fam in seen:
            continue
        seen.add(fam)
        picks.append({
            "repo": repo,
            "downloads": downloads,
            "quant": best["quant"],
            "size_gb": size_gb,
            "files": best["files"],
        })
    print("\r" + " " * 72 + "\r", end="")   # clear the scan line
    return picks, gated_hit, errors


# ---------------------------------------------------------------- download
def human(x):
    return f"{x:.2f} GB" if x >= 1 else f"{x * 1024:.0f} MB"


def download_file(url, dest_path, token=None, chunk=1024 * 1024):
    """Stream url -> dest_path with one-line % progress. Resumable via .part."""
    tmp = dest_path + ".part"
    headers = auth_headers(token)
    done = 0
    mode = "wb"
    if os.path.isfile(tmp):                     # resume support
        done = os.path.getsize(tmp)
        headers["range"] = f"bytes={done}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:                       # .part already complete
            os.replace(tmp, dest_path)
            return dest_path, 0
        if e.code in (401, 403):
            raise RuntimeError("401/403: gated repo. Put a read token in "
                               "hf_config.json ('hf_token').") from e
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e}") from e

    total = int(resp.headers.get("Content-Length") or 0) + done
    resumed = done > 0 and resp.status == 206
    if not resumed and done:
        done = 0
        mode = "wb"
    start = time.perf_counter()
    with open(tmp, mode) as f:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                pct = 100.0 * done / total
                rate = done / max(1e-9, time.perf_counter() - start)
                eta = (total - done) / rate if rate > 0 else 0
                rate_s = (f"{rate / GB:,.1f} GB/s" if rate >= 0.1 * GB
                          else f"{rate / (1024 ** 2):,.0f} MB/s")
                line = (f"    {pct:5.1f}%  {human(done / GB)} / "
                        f"{human(total / GB)}  {rate_s}  "
                        f"eta {int(eta // 60)}:{int(eta % 60):02d}   ")
                print("\r" + line, end="", flush=True)
    print("\r" + " " * 72 + "\r", end="")       # clear the progress line
    resp.close()
    os.replace(tmp, dest_path)
    return dest_path, done


def offer_download(picks, model_dir, token):
    print(f"\nDownload one of these into: {model_dir}")
    try:
        choice = input("Pick 1-5 to download, or press Enter to quit: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    if not choice or not choice.isdigit() or not (1 <= int(choice) <= len(picks)):
        print("Nothing downloaded. Re-run any time.")
        return
    pick = picks[int(choice) - 1]
    publisher, _, rname = pick["repo"].partition("/")
    dest_dir = os.path.join(model_dir, publisher, rname)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\nDownloading {pick['quant']} ({len(pick['files'])} file(s), "
          f"~{pick['size_gb']:.2f} GB) from {pick['repo']} ...")
    ok = True
    for fname in pick["files"]:
        base = os.path.basename(fname)
        dest = os.path.join(dest_dir, base)
        url = (f"{HF}/{pick['repo']}/resolve/main/"
               f"{urllib.parse.quote(fname)}")
        print(f"  -> {base}")
        try:
            final, _ = download_file(url, dest, token=token)
        except RuntimeError as e:
            print(f"\n  FAILED: {e}")
            ok = False
            break
        print(f"  saved: {final}")
    if ok:
        print("\nDone. In LM Studio / Bionic: open the models directory and "
              "rescan, then load the new model.")


# -------------------------------------------------------------------- main
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=None,
                    help="override usable memory budget in GB")
    ap.add_argument("--dir", default=None,
                    help="override destination model directory")
    ap.add_argument("--list-only", action="store_true",
                    help="show the 5 picks and skip the download prompt")
    args = ap.parse_args()

    cfg, token = load_config()

    # 1) Hardware ------------------------------------------------------------
    snap = llm_hw.snapshot()
    ram = snap.get("ram_bytes")
    print("=" * 78)
    print("MACHINE SUMMARY")
    print("=" * 78)
    print(f"OS          : {snap['os']} ({snap['machine']})")
    print(f"CPU         : {snap['cpu']}  ({snap['cpu_cores']} threads)")
    print(f"System RAM  : {ram / GB:.1f} GB total, "
          f"{(snap.get('avail_ram_bytes') or 0) / GB:.1f} GB available")

    nv_names = set()
    for g in snap.get("nvidia_gpus", []):
        nv_names.add(g["name"].lower())
        print(f"NVIDIA GPU  : {g['name']} - {g['vram_bytes'] / GB:.1f} GB VRAM "
              f"(sm_{g['compute_cap'].replace('.', '')})")

    amd_names = set()
    for a in snap.get("amd_gpus", []):
        amd_names.add(a["name"].lower())
        vram = f"{a['vram_bytes'] / GB:.1f} GB" if a.get("vram_bytes") else "n/a"
        print(f"AMD GPU     : {a['name']} - {vram}")

    for a in snap.get("display_adapters", []):
        low_name = a["name"].lower()
        if low_name in nv_names or low_name in amd_names:
            continue
        vram = f"{a['vram_bytes'] / GB:.1f} GB" if a.get("vram_bytes") else "n/a"
        print(f"Adapter     : [{a['vendor']}] {a['name']} - {vram}")

    apple = snap.get("apple")
    if apple:
        print(f"Apple chip  : {apple['chip']}"
              + (f", {apple['gpu_cores']} GPU cores" if apple.get("gpu_cores") else ""))
    budget = args.budget
    if budget is None:
        budget = cfg.get("budget_gb_override")
    if budget is None:
        budget, why = llm_hw.detect_budget()
    else:
        why = "--budget flag"
    print(f"LLM budget  : ~{budget:.1f} GB usable ({why})")

    # 2) Discover ------------------------------------------------------------
    print()
    print(f"Searching Hugging Face for models that FIT this machine "
          f"(budget ~{budget:.1f} GB) ...")
    picks, gated_hit, errors = discover_models(cfg, token, budget)
    if not picks:
        print("No fitting GGUF model found in the top results. Try "
              "--budget or raise 'scan_limit' in hf_config.json.")
        return
    if gated_hit and not token:
        print("  (Some repos are gated and were skipped - see hf_config.json "
              "to add a token.)")
    if errors:
        print(f"  ({errors} repos could not be inspected and were skipped.)")

    # Bigger model that still FITS = better pick (same rule as the probe).
    picks.sort(key=lambda p: -p["size_gb"])

    print()
    print(f"{'#':<3}{'Repo':<52}{'Quant':<9}{'Size':>9}  Downloads")
    print("-" * 92)
    for i, p in enumerate(picks, 1):
        print(f"{i:<3}{p['repo'][:50]:<52}{p['quant']:<9}"
              f"{p['size_gb']:>7.2f}GB  {p['downloads']:,}")
    print()
    print("Verdicts: every pick fully fits the budget (weights + ~15% runtime/KV).")
    print("Higher size at the same fit = better quality quant.")

    # Engine guidance for AMD systems (budget math assumes the VRAM is usable).
    amd_discrete = [a for a in snap.get("amd_gpus", [])
                    if (a.get("vram_bytes") or 0) >= 4 * GB]
    if amd_discrete and "nvidia" not in str(snap.get("budget_source", "")).lower():
        import platform as _pf
        if _pf.system() == "Linux":
            print("AMD GPU detected: use Ollama's ROCm build, or llama.cpp with")
            print("the Vulkan or ROCm (HIP) backend so the model runs in VRAM.")
        elif _pf.system() == "Windows":
            print("AMD GPU detected: use the Vulkan backend (llama.cpp Vulkan build,")
            print("LM Studio's Vulkan runtime). Ollama's ROCm build supports recent")
            print("Radeon cards (RX 7000/9000 series) on Windows as well.")

    # 3) Download ------------------------------------------------------------
    model_dir = args.dir or resolve_model_dir(cfg)
    if args.list_only:
        print(f"\n(--list-only) Download target would be: {model_dir}")
        return
    offer_download(picks, model_dir, token)


if __name__ == "__main__":
    main()
