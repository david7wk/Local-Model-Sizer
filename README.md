# Local-Model-Sizer

**Find, size, and fetch open-weight LLMs that actually run on *your* machine —
in one command, with zero pip installs.**

`Local-Model-Sizer` started as a sizing methodology for local ("open weight")
LLMs on consumer hardware and grew into working tooling. Its flagship tool,
**`model_fetcher`**, closes the loop end-to-end:

```
probe the machine ──> memory budget (GB) ──> search Hugging Face (GGUF)
      ──> pick the best quant that FITS, per model ──> show the 5 best
      ──> you choose one ──> download with live % progress
      ──> into ~/.lmstudio/models/<publisher>/<repo>/<file>.gguf
```

The result lands where **LM Studio** and **Bionic** (which is built on LM
Studio) can load it immediately, and works equally with **Ollama** and
**llama.cpp**.

- **Platform independent** — Python 3.8+ **standard library only** (no pip
  packages, no compilers). Developed and tested on Windows 11 and designed
  for macOS (Apple Silicon incl. M5) and Linux.
- **GPU-vendor aware** — NVIDIA (CUDA), AMD Radeon (ROCm/Vulkan), and Apple
  Silicon unified memory all feed the budget math, with per-vendor engine
  guidance.

## Repository layout

| Path | What it is |
|---|---|
| [`model_fetcher/`](model_fetcher/) | The tool. **Start at [`model_fetcher/README.md`](model_fetcher/README.md)** — full setup (venv), options, captured example runs, download destination, and ecosystem notes. |
| [`model_fetcher/DEVELOPMENT_LOG.md`](model_fetcher/DEVELOPMENT_LOG.md) | Project chronicle: inception, design decisions, session-by-session progress, roadmap. |

## Quick start

```powershell
cd model_fetcher
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt    # no-op: the tool is stdlib-only
python hf_recommend_and_fetch.py --list-only
```

Typical flow:

```bash
python hf_recommend_and_fetch.py                  # probe -> 5 picks -> download prompt
python hf_recommend_and_fetch.py --budget 5.2     # simulate a smaller machine
python hf_recommend_and_fetch.py --dir "D:/LLM/models"
```

Downloads default to `~/.lmstudio/models/<publisher>/<repo>/<file>.gguf`
(override with `--dir` or `hf_config.json`). A Hugging Face token is
**optional** — only gated repos (Gemma, Llama, …) need one; put it in
`hf_config.json` (auto-created on first run) and keep that file private.

## How it decides what "fits"

Usable budget = GPU VRAM / Apple unified memory minus an OS reserve. A GGUF
file **FITS** when `file size × 1.15 ≤ budget` (the +15% covers runtime and
KV-cache overhead). For each candidate repo the tool selects the
*highest-quality* quant that FITS (never below Q3/IQ3), rejects draft/MTP
modules masquerading as full models, dedupes model families, and ranks the
final five largest-model-first — the same "pick the largest model that FITS"
rule used by the companion probe tooling.

## Status & roadmap

See [`model_fetcher/DEVELOPMENT_LOG.md`](model_fetcher/DEVELOPMENT_LOG.md) for
the full history, design decisions (D1–D5), and the prioritized roadmap —
next up: SHA256 download verification, `--json` output, an `--mlx` mode for
Apple Silicon, and Ollama auto-registration.

## License

See [LICENSE](LICENSE).
