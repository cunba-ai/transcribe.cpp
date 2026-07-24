# Upgrading CUDA & GPU Architectures in CI

A practical guide for changing the CUDA toolkit version and/or the set of
GPU compute capabilities (`sm_*`) a CI pipeline compiles for. Written from
a real upgrade (CUDA 12.4 → 12.9, adding Blackwell `sm_120`) that hit four
non-obvious version-compatibility traps. Use it as a checklist so the next
project only has to learn these once.

---

## The short version

Upgrading CUDA is **not** "bump one version number." Four things must be
consistent, and each comes from a *different source* with a *different*
version list:

| Component | Source | Where its version list lives |
|-----------|--------|------------------------------|
| Linux CUDA toolkit | Docker Hub `nvidia/cuda` image | https://hub.docker.com/r/nvidia/cuda/tags |
| Windows CUDA toolkit | `Jimver/cuda-toolkit` action | https://github.com/Jimver/cuda-toolkit/releases |
| GPU arch support floor | The CUDA toolkit itself | https://docs.nvidia.com/cuda/blackwell-compatibility-guide/ |
| `CMAKE_CUDA_ARCHITECTURES` spelling | CMake + nvcc | https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ |

**The four traps, in order of how likely they bite you:**

1. **`Jimver/cuda-toolkit` has gaps in its version list.** It never shipped
   12.8.x at all (jumps 12.9.0 → 13.0.0). A version that exists on Docker Hub
   may not exist in the action. Always cross-check.
2. **New GPU architectures have a CUDA floor.** `sm_120` (Blackwell, RTX 50)
   needs CUDA **≥ 12.8**. Compiling for it on 12.4/12.6 fails at configure
   or build. The floor is per-architecture, not per-toolkit-major.
3. **Action version pins rot independently of CUDA versions.** An old action
   tag (`v0.2.21`) may not know about newer CUDA releases even if the action
   *could* install them. Bump the action tag too.
4. **Linux and Windows toolkits come from different sources** and may not
   offer the *same* patch version. Align on a **series** (e.g. 12.9.x), not
   an exact patch — a 0.0.x difference is irrelevant for arch support.

---

## Step-by-step checklist

### 1. Decide the architecture list

Map each `sm_*` you want to its CUDA floor and the GPUs it covers:

| Arch | Codename | Representative GPUs | Min CUDA |
|------|----------|---------------------|----------|
| 60 | Pascal | GTX 10-series low | 8.0 |
| 61 | Pascal | GTX 1080/1070 | 8.0 |
| 70 | Volta | TITAN V, V100 | 9.0 |
| 75 | Turing | RTX 20-series, T4 | 10.0 |
| 80 | Ampere | A100 | 11.0 |
| 86 | Ampere | RTX 30-series, A40 | 11.1 |
| 89 | Ada Lovelace | RTX 40-series | 11.8 |
| 90 | Hopper | H100 (datacenter) | 11.8 |
| 120 | Blackwell | RTX 50-series | **12.8** |

Pick your floor from the **max** of the architectures you keep. If you keep
`sm_120`, your toolkit floor is 12.8 — no exceptions.

The `CMAKE_CUDA_ARCHITECTURES` spelling is the bare number (`120`, not
`sm_120` or `12.0`). Confirm at
https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

### 2. Pick the CUDA toolkit series

Choose the lowest series that (a) meets your arch floor and (b) is available
in **both** sources below. Don't assume a version exists in one just because
it exists in the other.

**Linux** — query Docker Hub's API for the exact tag:

```bash
# Does a devel image (with nvcc) exist for this series + ubuntu?
curl -s "https://hub.docker.com/v2/repositories/nvidia/cuda/tags?page_size=100&name=12.9-devel-ubuntu22.04" \
  | python3 -c "import sys,json; [print(t['name']) for t in json.load(sys.stdin)['results']]"
```

You want the `-devel-` variant (it carries `nvcc`); `-runtime-` does not.

**Windows** — check the action's release notes for which CUDA versions each
tag added. As of writing, the gaps are:

```
Jimver/cuda-toolkit version history (CUDA versions ADDED):
  v0.2.24 → 12.9.0
  v0.2.26 → 12.9.1 (link update)
  v0.2.27 → 13.0.0
  v0.2.28 → 13.0.1
  v0.2.29 → 13.0.2
  v0.2.30 → 13.1.0
  v0.2.32 → 13.1.1, 13.2.0
  v0.2.35 → (node24, no new CUDA)
  *** 12.8.x NEVER EXISTED in this action ***
```

Pick the action tag that (a) lists your CUDA version and (b) is recent
enough to not trip the Node.js deprecation warning (use ≥ the tag that
shipped node20, and prefer the latest for node24).

### 3. Find every place the old values live

CUDA version and arch lists are usually repeated in 3–4 spots. Grep them all
**before** editing, or you'll ship an inconsistent build:

```bash
# In this repo's CI + build config:
rg -n "CUDA_ARCHS|CMAKE_CUDA_ARCHITECTURES|nvidia/cuda:|cuda-toolkit@|cuda: '" \
  .github/workflows/ CMakeLists.txt CMakePresets.json pyproject.toml scripts/
```

Typical locations:
- Workflow `env:` block (a shared `CUDA_ARCHS` the matrix references)
- Each Linux matrix entry's `container:` and `cmake_extra:`
- The Windows `Setup CUDA Toolkit` step (`uses:` and `cuda:`)
- `CMakePresets.json`, per-binding `pyproject.toml`, local build scripts

**Decide scope deliberately.** The C ABI pipeline, the Python wheel
provider, and local dev scripts are independent build paths — you don't
have to move them all at once, but record which ones you touched and which
you left.

### 4. Make the edits

For a GitHub Actions matrix build, the four edits are:

```yaml
env:
  # (1) shared arch list, referenced by Windows configure
  CUDA_ARCHS: "61;70;75;80;86;89;120"

jobs:
  linux:
    matrix:
      config:
        - name: CUDA-CPU
          # (2) Linux: devel image (has nvcc), ubuntu version must match base
          container: nvidia/cuda:12.9.1-devel-ubuntu22.04
          cmake_extra: '-DCMAKE_CUDA_ARCHITECTURES="61;70;75;80;86;89;120"'
  windows:
    steps:
      - uses: Jimver/cuda-toolkit@v0.2.35    # (3) action tag that HAS your CUDA
        with:
          cuda: '12.9.0'                       # (4) version the action actually ships
```

Note Linux is `12.9.1` (Docker Hub) and Windows is `12.9.0` (the action) —
**same series, different patch, both fine.** Don't force them identical;
force them same-series.

### 5. Verify before pushing

```bash
# YAML still parses
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/<file>.yml'))"

# No stale references to the old versions remain
rg -n "<old-cuda-version>|<old-arch-list>" .github/workflows/   # expect empty
```

### 6. Watch the first run for the three failure signatures

| Symptom | Meaning | Fix |
|---------|---------|-----|
| `Error: Version not available: X.Y.Z` (Windows, <30s in) | Action doesn't ship that version | Pick a version the action's release notes list; bump action tag |
| configure error: `unsupported CUDA architecture` / `Unknown CUDA arch` | Toolkit too old for an arch in your list | Raise the toolkit series to that arch's floor |
| `nvcc fatal: Unsupported gpu architecture 'compute_XX'` | Wrong `CMAKE_CUDA_ARCHITECTURES` spelling | Use the bare number (`120`), confirm against the gencode table |

The "Version not available" failure is **instant** (the action aborts before
any build) — so push and watch the Windows job for the first 30 seconds; if
it clears Setup CUDA Toolkit, the version pairing is good and the rest is
just compile time.

---

## Worked example: this repo's upgrade

**From** CUDA 12.4.1 (Linux) / 12.6.2 (Windows), archs
`60;61;70;75;80;86;89;90`.

**Goal** add Blackwell `sm_120`, drop the oldest/newest-datacenter ends.

**Path taken (with the detour):**

1. First attempt: set everything to **12.8.1**, archs
   `61;70;75;80;86;89;120`.
2. Linux container `nvidia/cuda:12.8.1-devel-ubuntu22.04` — **existed and
   configured fine.**
3. Windows `Jimver/cuda-toolkit@v0.2.21` with `cuda: '12.8.1'` — **failed
   in 22 seconds**: `Error: Version not available: 12.8.1`. The action never
   shipped 12.8.x.
4. Correction: Windows to `cuda: '12.9.0'` + `@v0.2.35` (latest, node24);
   Linux aligned to the same 12.9 series (`12.9.1-devel`).
5. Both configured and built successfully.

**Lesson:** the Docker image existed for 12.8 but the action did not. Had
we checked the action's version list *first* (step 2 above), we'd have gone
straight to 12.9 and skipped the failed run.

---

## Why the toolkit version matters beyond "does it compile"

Two downstream effects worth knowing:

**fatbin size is roughly linear in arch count.** Each `sm_*` adds ~150–180MB
of PTX+SASS to the `.nv_fatb` / `.nv_fatbin` section. Going from 8 archs to 7
shrinks the fatbin by ~12–15%, not 50%. If artifact size is the goal,
*removing redundant static archives* and *preserving symlinks in the zip*
are far bigger levers than trimming archs — see the C ABI build workflow's
collect-artifacts step for both.

**`-devel` vs `-runtime` images.** Only `-devel-` images contain `nvcc`, so
only they can *compile* CUDA. `-runtime-` images can only *run* already-
compiled CUDA binaries. A build pipeline always needs `-devel-`; a deploy
image can use `-runtime-` to stay smaller.

---

## Quick reference: where to check each thing

- **Does Docker Hub have this nvidia/cuda tag?**
  `https://hub.docker.com/v2/repositories/nvidia/cuda/tags?name=<version>-devel-ubuntu<xx.04>`
- **Does the cuda-toolkit action ship this CUDA version?**
  `https://github.com/Jimver/cuda-toolkit/releases` (read the "Added" lines)
- **What's the CUDA floor for an arch?**
  `https://docs.nvidia.com/cuda/blackwell-compatibility-guide/` (and per-arch
  NVIDIA docs)
- **What's the CMAKE_CUDA_ARCHITECTURES spelling?**
  `https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/`
- **What GPUs map to which sm_*?**
  Same gencode table, or `https://developer.nvidia.com/cuda-gpus`
