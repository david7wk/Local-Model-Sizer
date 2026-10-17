# hf_recommend_and_fetch.py

**Find, size, and download local-LLM models that actually run on *this* machine.**

A platform-independent Python script (Python 3.8+ **standard library only** —
no pip packages, no compilers) that:

1. **Probes the machine** — OS, CPU, system RAM, GPU VRAM (NVIDIA, AMD Radeon,
   Apple Silicon unified memory) and computes a usable *LLM memory budget*.
2. **Searches Hugging Face** for popular GGUF models and computes which
   quantization of each truly **FITS** that budget (weights + ~15% runtime/KV
   allowance), then lists the **5 best picks**.
3. **Downloads your pick** with a single-line percentage progress display
   (resumable) into a model directory that **Bionic and LM Studio** read.

```
probe machine ──> budget (GB) ──> search GGUF repos ──> pick best quant
      per repo that FITS ──> present 5 ──> user chooses ──> download
      into ~/.lmstudio/models/<publisher>/<repo>/<file>.gguf
```

---

## Setup

### 1. Create a virtual environment (recommended)

**Windows 11 (PowerShell):**

```powershell
cd scripts\model_fetcher
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**

```bash
cd scripts/model_fetcher
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

This is intentionally a **no-op**: the script uses only the Python standard
library (`urllib`, `json`, `re`, `ctypes`, …). The requirements file documents
this and lists the optional `huggingface_hub` dependency used by the companion
script `../crossplat/hf_model_sizer.py`, should you enable it there.

> If `llm_hw.py` is missing from this folder the script exits with a clear
> message — it is the shared hardware-detection module and ships with this
> folder (master copy lives in `../crossplat/llm_hw.py`).

---

## Configuration — `hf_config.json`

Auto-created on first run. All fields optional:

| Key | Default | Meaning |
|---|---|---|
| `hf_token` | `""` | **Optional.** Hugging Face *read* token (create free at <https://huggingface.co/settings/tokens>). Only needed for **gated** repos (Gemma, Llama, …). The `HF_TOKEN` environment variable is honored as a fallback. |
| `model_dir` | `""` | Download destination. Empty = `~/.lmstudio/models` (see below). |
| `budget_gb_override` | `null` | Force a memory budget instead of auto-detection. |
| `max_results` | `5` | How many candidate models to list. |
| `scan_limit` | `40` | How many HF search results to evaluate per search. |

Keep this file private once you add a token (it is plain text).

---

## Command-line options

| Option | Meaning |
|---|---|
| *(none)* | Interactive: probe → list 5 picks → download prompt. |
| `--budget GB` | Override the auto-detected memory budget (e.g. `--budget 5.2` to *simulate* a smaller machine). |
| `--dir PATH` | Override the download destination for this run (beats `model_dir` in the config). |
| `--list-only` | Show the picks and skip the download prompt. |

Typical invocations:

```bash
python hf_recommend_and_fetch.py                  # full interactive flow
python hf_recommend_and_fetch.py --list-only      # just see the 5 picks
python hf_recommend_and_fetch.py --budget 5.2     # simulate a 6 GB laptop
python hf_recommend_and_fetch.py --dir "D:/LLM/models"
```

---

## Example runs (captured output)

### A. Full flow on a Lenovo Legion 7a (RTX 5070 Laptop, 12 GB) — `--list-only`

```text
==============================================================================
MACHINE SUMMARY
==============================================================================
OS          : Windows 11 (AMD64)
CPU         : AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD  (24 threads)
System RAM  : 31.3 GB total, 6.7 GB available
NVIDIA GPU  : NVIDIA GeForce RTX 5070 Laptop GPU - 11.9 GB VRAM (sm_120)
Adapter     : [AMD] AMD Radeon(TM) 890M Graphics - 0.5 GB
Adapter     : [NVIDIA] NVIDIA GeForce RTX 5070 Laptop GPU - 11.9 GB
LLM budget  : ~11.1 GB usable (NVIDIA VRAM (NVIDIA GeForce RTX 5070 Laptop GPU) minus 0.8 GB reserve)

Searching Hugging Face for models that FIT this machine (budget ~11.1 GB) ...

  scanning Hugging Face... 16/80  (4 fits so far)

#  Repo                                                Quant         Size  Downloads
--------------------------------------------------------------------------------------------
1  ibm-granite/granite-4.2-8b-GGUF                     Q8_0        8.70GB  258,458
2  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF                 Q8_0        7.54GB  321,014
3  ibm-granite/granite-4.2-3b-GGUF                     BF16        6.82GB  257,188
4  AnkitAI/Parable-Granite-4.1-3B-Claude-Fable-5-GGUF  F16         6.34GB  180,380
5  Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF               Q8_0        1.76GB  206,081

Verdicts: every pick fully fits the budget (weights + ~15% runtime/KV).
Higher size at the same fit = better quality quant.

(--list-only) Download target would be: C:\Users\david\.lmstudio\models
```

Notice how each repo is represented by the **highest-quality quant that FITS**
(Q8_0 of an 8B here), not merely the first file found.

### B. Simulating a smaller machine — `--budget 5.2 --list-only`

Same hardware, budget forced to 5.2 GB (e.g. planning for a 6 GB laptop):

```text
LLM budget  : ~5.2 GB usable (--budget flag)

#  Repo                                                Quant         Size  Downloads
--------------------------------------------------------------------------------------------
1  ibm-granite/granite-4.2-8b-GGUF                     Q3_K_L      4.38GB  258,458
2  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF                 Q4_0        4.13GB  321,014
3  ibm-granite/granite-4.2-3b-GGUF                     Q8_0        3.63GB  257,188
4  AnkitAI/Parable-Granite-4.1-3B-Claude-Fable-5-GGUF  Q8_0        3.37GB  180,380
5  Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF               Q8_0        1.76GB  206,081
```

Same model families, but the quants dropped (Q3_K_L / Q4_0 instead of Q8_0) —
exactly the trade the FITS math is designed to make visible.

### C. Interactive download with single-line progress

When run without `--list-only`, the script ends with:

```text
Download one of these into: C:\Users\david\.lmstudio\models
Pick 1-5 to download, or press Enter to quit: 3

Downloading Q8_0 (1 file(s), ~3.63 GB) from ibm-granite/granite-4.2-3b-GGUF ...
  -> granite-4.2-3b-Q8_0.gguf
     12.5%  0.45 GB / 3.63 GB  28 MB/s  eta 2:14   ← live, redraws on ONE line
    100.0%  3.63 GB / 3.63 GB  31 MB/s  eta 0:00
  saved: C:\Users\david\.lmstudio\models\ibm-granite\granite-4.2-3b-GGUF\granite-4.2-3b-Q8_0.gguf

Done. In LM Studio / Bionic: open the models directory and rescan, then load the new model.
```

(Real capture from a 1.71 GB download on this machine; the progress line
updates in place via carriage-return redraw, with percent, bytes, throughput,
and ETA. Interrupted downloads resume from a `.part` file.)

---

## Download destination & the local-LLM ecosystem

**Default destination:** `~/.lmstudio/models` — the model root used by
**LM Studio** and by **Bionic** (which is built on LM Studio). Inside it, the
script reproduces the layout LM Studio expects:

```
~/.lmstudio/models/
└── ibm-granite/                       # publisher (HF author)
    └── granite-4.2-3b-GGUF/           # repo name
        └── granite-4.2-3b-Q8_0.gguf   # the quant file you picked
```

After a download, open LM Studio (or Bionic) → *My Models* → **rescan**, and
the model appears ready to load.

**Why GGUF?** It is the one format the whole local ecosystem consumes:

| Runtime | Works with the downloaded GGUF via |
|---|---|
| **LM Studio / Bionic** | native (this is the default destination) |
| **Ollama** | `ollama pull hf.co/<publisher>/<repo>:<QUANT>` or a Modelfile with `FROM` pointing at the .gguf |
| **llama.cpp / llama-server** | directly (`-m path/to/model.gguf`) |

**Engine backends per GPU vendor** (the script's budget math already assumes
the VRAM is usable, and prints these hints when relevant):

- **NVIDIA** — CUDA builds of Ollama / llama.cpp (laptops: `nvidia-smi` is
  located even when it is not on PATH).
- **AMD Radeon** — Linux: Ollama ROCm or llama.cpp Vulkan/ROCm (HIP).
  Windows: llama.cpp Vulkan build, LM Studio's Vulkan runtime; Ollama's ROCm
  build supports recent Radeon cards (RX 7000/9000). Discrete cards ≥ 4 GB
  drive the budget; APUs/iGPUs report only a small dedicated carve-out and are
  conservatively excluded.
- **Apple Silicon (M1–M5)** — unified memory minus a reserve defines the
  budget; GGUF runs via llama.cpp/Ollama Metal, while `mlx-community/*`
  builds are often faster inside LM Studio. (This script fetches GGUF; use
  `../crossplat/hf_model_sizer.py --format mlx` to size MLX repos.)

---

## How the 5 picks are chosen

1. Candidate repos come from **curated general-purpose LLM families**
   (Qwen3, Llama-3.x, Gemma-3, Mistral, Phi-4, gpt-oss, Qwen2.5-Coder,
   SmolLM3, Granite), each searched by download popularity — a raw
   "most downloaded GGUF" query surfaces audio/embedding/misc repos.
2. For every repo the script reads **real file sizes** from the HF API,
   groups sharded files ( `-00001-of-00003.gguf` …) into complete quant sets,
   and keeps the **largest quant whose size × 1.15 fits the budget**.
3. Quality gates remove noise: quants below Q3/IQ3 (too degraded for chat),
   full-precision files under 2 GB (draft/MTP modules, not real models), and
   files whose size is impossible for the parameter count in the repo name
   (e.g. a 0.86 GB "F16" in a 27B repo).
4. Results are deduplicated by model family and ranked **largest model first**
   — the same "pick the largest model that FITS" rule as the probe.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing helper module: llm_hw` | Keep `llm_hw.py` in the same folder as the script. |
| `HTTP 401/403 ... gated repo` | Paste a free HF read token into `hf_config.json` (`hf_token`). |
| Download stalls / interrupted | Re-run and pick the same model — the `.part` file resumes where it stopped. |
| Model doesn't appear in LM Studio | Use *My Models → Rescan*, or check the folder layout matches `publisher/repo/file.gguf`. |
| No GPU budget detected (falls back) | On AMD, install `rocm-smi` (Linux) or check the driver; on Windows laptops the NVIDIA driver's `nvidia-smi` is found automatically. Override with `--budget` anytime. |
| Picks are all too small/big | Tune with `--budget`, or raise `scan_limit` in `hf_config.json` to evaluate more repos. |
