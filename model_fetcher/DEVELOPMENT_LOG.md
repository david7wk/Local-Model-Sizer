# model_fetcher — Development Log & Project Chronicle

> Living document. One dated entry per feature or change from here forward.
> When you push to the new GitHub repo, each entry below maps naturally to a
> release note / commit message.

- **Project:** `model_fetcher` — find, size, and fetch local-LLM models that
  actually run on the target machine.
- **Inception:** 2026-10-14 (evening session, LM Studio Bionic agent)
- **Parent project:** *Local LLM Open-Weight Model Sizing and Use for select
  ERC Systems* — this tool operationalizes the model-sizing guidance from
  `Choosing-Optimal-Local-LLM-Models-ERC-Dev-Systems.md` and the
  `Modelfiles/` collection.
- **Language/runtime:** Python 3.8+ standard library only (deliberate — see
  Decision D1).
- **Target machines:** MacBook Pro (Apple M5), Razer Blade 15 (Windows 11),
  Lenovo Legion 7a (Windows 11, RTX 5070 Laptop + Radeon 890M iGPU).

---

## 1. Inception — why this tool exists

The parent project established a sizing methodology (memory budget =
VRAM/unified memory minus OS reserve; model file + ~15% runtime/KV allowance;
"pick the largest model that FITS"). Three utility scripts existed to apply it
by hand:

| Script | Role |
|---|---|
| `hf_model_sizer.py` | Inspect a HF repo's quants vs a budget (FIT/TIGHT/STRETCH) |
| `llm_hardware_probe.py` | Probe hardware, print model-class recommendations |
| `ollama_bench.py` | Measure TTFT / prompt tok/s / gen tok/s + GPU placement |

**Gap identified tonight:** the workflow was still manual and per-machine —
nothing *combined* "what does this machine have" → "what should I download"
→ "put it where my local-LLM runtime can use it." `hf_recommend_and_fetch.py`
was conceived to close that loop in one command.

---

## 2. Session log

### 2026-10-14 — Session 1: Cross-platform v2 of the tool suite

**Goal:** make the three original scripts genuinely portable across the three
target laptops.

**Delivered** (`scripts/crossplat/`):

- **`llm_hw.py`** — new shared hardware-detection module (stdlib only):
  - `nvidia-smi` **path auto-discovery** (PATH → `where` → `%WINDIR%\System32`
    → `Program Files\NVIDIA Corporation\NVSMI`) — fixes laptops where the
    driver dir is not on PATH.
  - Windows **WMI fallback** (`Win32_VideoController`) + registry
    `HardwareInformation.qwMemorySize` VRAM refinement (WMI `AdapterRAM` is
    capped at ~4 GB).
  - Apple Silicon enrichment: chip name/variant, P-cores, GPU core count via
    `ioreg -rc AGXAccelerator` (`sysctl hw.gpucores` does **not** exist),
    documented memory-bandwidth table, M5 awareness.
  - Single `detect_budget()` shared by all scripts.
- **`hf_model_sizer.py` v2, `llm_hardware_probe.py` v2** — same CLIs, budget
  detection delegated to `llm_hw`; probe adds hybrid-graphics laptop guidance
  (Optimus/MUX, Lenovo Vantage dGPU-only note) and `--json`.
- **`ollama_bench.py` v2** — adds `--host`/`OLLAMA_HOST` (bench a remote box),
  host normalization (`0.0.0.0` → `127.0.0.1`), median/spread stats, `--json`.

**Verified on the Legion 7a:** probe detected the RTX 5070 Laptop GPU
(sm_120, 11.9 GB) → ~11.1 GB budget; correct FITS/STRETCH verdicts;
hybrid-graphics section fired.

### 2026-10-14 — Session 2: `hf_recommend_and_fetch.py` created

**Goal:** one command: probe → 5 fitting models → download the chosen one.

**Key design decisions:**

- **D1 — stdlib-only HF access.** `huggingface_hub` could not install in the
  dev sandbox (Python 3.15; no PyYAML wheel available) and is a recurring
  friction point on fresh machines. Implemented the HF REST API
  (`/api/models?filter=gguf&sort=downloads`, `/api/models/{repo}?blobs=true`)
  and a streaming downloader directly with `urllib`. Consequence: **zero pip
  installs** to run the tool on any of the three laptops.
- **D2 — LM Studio layout as the destination contract.**
  `~/.lmstudio/models/<publisher>/<repo>/<file>.gguf` — consumed natively by
  LM Studio and Bionic; importable by Ollama and llama.cpp (see README).
- **D3 — verdict math inherited from the sizer** (weights × 1.15 ≤ budget ⇒
  FITS), moved into `llm_hw.py` in Session 3 so there is one source of truth.
- **D4 — quality gates over raw popularity.** First live run exposed that
  "most-downloaded GGUF" surfaces non-LLM repos (bird classifier, audio
  models) and junk files. Gates added:
  1. Curated family searches (Qwen3, Llama-3.x, Gemma-3, Mistral, Phi-4,
     gpt-oss, Qwen2.5-Coder, SmolLM3, Granite) instead of one raw query.
  2. Reject quants below Q3/IQ3 (chat quality floor); `IQ1` added to the
     quant-rank table after `IQ1_M` ranked as *top* quality.
  3. Reject full-precision (F16/BF16/F32) files under 2 GB (draft/MTP/LoRA
     masquerading as full models).
  4. Size-vs-params plausibility check (a 0.86 GB "F16" in a 27B repo is not
     the full model).
  5. Shard groups must be complete (`-0000i-of-0000N` count == N) and
     single-file variants preferred over their shard duplicates.
  6. Repos with unrecognizable quant labels are skipped (can't be ranked).
- **D5 — resumable, single-line progress.** 1 MB chunk streaming with
  carriage-return redraw (percent / bytes / throughput / ETA), HTTP Range
  resume via `.part` files — multi-GB GGUFs survive flaky Wi-Fi.

**Bugs found and fixed during live testing** (proof the pipeline works):

| Bug | Root cause | Fix |
|---|---|---|
| HTTP 400 on every repo inspection | repo id was URL-encoded *including* the `/` (`%2F`) rejected by HF | quote owner/name separately, keep `/` literal |
| `IQ1_M` chosen as "best" quant | not in `QUANT_RANK` → ranked last→first as unknown | `IQ1` added at rank 0 in `llm_hw.QUANT_RANK` |
| lowercase files (`q4_k_m.gguf`) grouped wrong | shard key and single-file key differed in case/`.gguf` suffix | normalized group keys; single-file preferred |
| 0.86 GB "BF16/F16 27B" picks | no parameter token in repo name to check against | F-class-under-2 GB rule + params-plausibility check |

**End-to-end download test:** 1.71 GB granite Q3_K_M downloaded with live
progress, saved in the LM Studio layout, then cleaned up.

### 2026-10-14 — Session 3: Packaging + AMD support

- Tool moved to its own portable folder, **`scripts/model_fetcher/`**:
  script + `hf_config.json` + `requirements.txt` + a copy of `llm_hw.py`
  (folder is self-contained for copying to the other laptops; `crossplat/`
  remains the `llm_hw` master copy).
- `requirements.txt` added (documents stdlib-only design; optional
  `huggingface_hub` note for the companion sizer).
- **AMD support:**
  - `llm_hw.amd_gpus()`: Linux sysfs (`/sys/class/drm/card*/device`, PCI
    vendor `0x1002`, `mem_info_vram_total`), name via `rocm-smi
    --showproductname`; Windows via WMI/registry path.
  - `detect_budget()` AMD branch: discrete cards **≥ 4 GB** dedicated VRAM
    drive the budget; APUs/iGPUs excluded (they report only a carve-out).
  - Fetcher prints engine guidance for AMD (Linux: Ollama ROCm / llama.cpp
    Vulkan-HIP; Windows: Vulkan runtime, Ollama ROCm for RX 7000/9000).
- **Verified:** `amd_gpus()` lists the Legion's Radeon 890M; simulated
  RX 7900 XT (20 GB) resolves to a 19.2 GB "AMD VRAM … (Vulkan/ROCm engine)"
  budget; sizer still works after the quant-helper refactor.

### 2026-10-14 — Session 4: README for `model_fetcher/`

- venv creation (Windows PowerShell + bash), `pip install -r requirements.txt`
  (documented no-op), full CLI + config reference, three **captured** example
  runs (full machine, `--budget 5.2` simulation, download w/ progress),
  download-destination tree, ecosystem table (LM Studio / Bionic / Ollama /
  llama.cpp + per-vendor backends), pick-selection explanation, troubleshooting.

---

## 3. Current state (end of 2026-10-14)

**Files:**

```
scripts/model_fetcher/
├── hf_recommend_and_fetch.py   # the tool (stdlib only)
├── hf_config.json              # optional HF token, model_dir, budget override
├── requirements.txt            # documented no-op (stdlib design)
├── llm_hw.py                   # portable copy (master: scripts/crossplat/)
├── README.md                   # usage, captured examples, ecosystem notes
└── DEVELOPMENT_LOG.md          # this file

scripts/crossplat/              # v2 tool suite + master llm_hw.py + README
scripts/                        # original v1 scripts (left untouched)
```

**Feature status:**

| Feature | State |
|---|---|
| Hardware probe & budget | ✅ NVIDIA / AMD discrete / Apple Silicon / CPU fallback |
| 5-model discovery with FITS math | ✅ live-tested |
| Single-line resumable download | ✅ live-tested (1.71 GB) |
| Sharded-GGUF handling | ✅ |
| Gated-repo tokens via config | ✅ (code path in place; not exercised against a gated repo yet) |
| AMD budget + engine guidance | ✅ simulated + WMI path live |
| LM Studio/Bionic destination layout | ✅ |
| `--json` machine-readable output | ❌ not yet (probe/bench have it; fetcher doesn't) |

**Known limitations / future considerations:**

- APU-only machines (no dGPU) fall back to a RAM-based CPU budget; a
  shared-memory (GTT/"shared GPU memory") budget mode could be smarter.
- Curated family list is static; new model families need a one-line addition
  to `FAMILY_SEARCHES`.
- `family_key()` dedupe is heuristic; near-duplicate forks can still appear.
- No integrity check after download (HF offers SHA256 via the API — worth
  adding before public release).

---

## 4. Roadmap (proposed, in rough priority)

1. **GitHub repo bootstrap** — init repo, push `model_fetcher/` (plus
   `crossplat/` as the wider suite), add LICENSE + `.gitignore`
   (`.venv/`, `__pycache__/`, and **`hf_config.json`** if a token is ever
   committed by accident — better: ship `hf_config.example.json` and gitignore
   the real one).
2. **`--json` output** for the fetcher (parity with probe/bench; enables
   scripting/CI).
3. **SHA256 verification** of downloaded files using HF's blob metadata.
4. **`--mlx` mode** for Apple Silicon (pull `mlx-community` repos instead).
5. **Ollama auto-register** (generate a Modelfile, `ollama create`).
6. **APU shared-memory budget mode.**
7. **One-command pipeline runner** (sizer → probe → bench) across the suite.
8. Optional: `--model <repo>` direct mode (skip discovery, evaluate + fetch
   one repo).

---

## 5. Entry template (append below the line for each new feature)

```markdown
### YYYY-MM-DD — Session N: <feature title>

**Goal:** <one sentence>

**Delivered:** <bullets, file names>

**Design decisions (if any):** D# — <decision & rationale>

**Tests:** <what was run, on which machine, captured evidence>

**Known issues introduced/left open:** <list or "none">
```

---

*Chronicle started 2026-10-14. Entries above cover Sessions 1–4 (inception
night). Next entry: GitHub repo bootstrap (Roadmap item 1).*
