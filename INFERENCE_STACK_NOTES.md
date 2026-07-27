# Inference stack notes: LM Studio, llama.cpp, ROCm, Vulkan, CUDA

Date: 2026-07-27. Untracked on purpose while `require_clean_tree` runs are in
flight; commit it once the queue drains.

Operational reference for serving models in this project. Everything below was
measured or reproduced here, not taken from documentation.

## Speed: the three stacks are within 11%

Same RX 9070 XT, same `qwen3-14b Q4_K_M`, same 139 gold lines, loopback,
ctx 16384, f16 KV, `parallel 1`, idle card, all runs within 15 minutes:

| stack | s/call | relative |
|---|---:|---:|
| llama.cpp ROCm (b10121, HIP) | 0.1959 | 1.000 |
| llama.cpp Vulkan (b10142, RADV) | 0.2101 | 1.072 |
| LM Studio (ROCm) | 0.2173 | 1.109 |

Accuracy identical across all three (1-4 discordant lines per arm, all
p >= 0.63), which is the check that makes the timings meaningful - they are
doing the same work.

**LM Studio's GGUF runtime is llama.cpp**, so the spread is frontend overhead,
not a different engine. **ROCm beats Vulkan on RDNA4 by ~7%** - worth recording
because the common expectation is the reverse on new AMD hardware. Caveat: the
Vulkan package is 21 commits ahead (b10142 vs b10121), so that 7% is
stack + version, not stack alone.

Speed is not a reason to migrate.

## Capability: KV quantization is the real difference

`magistral-small` is recorded in the attribution brief as untested locally
because 13.35 GiB of weights would not fit 15.92 GiB safely. That assumes an
f16 KV cache:

| context | KV f16 | KV q8_0 | KV q4_0 |
|---|---:|---:|---:|
| 8192 | 14.60 | **13.97** | 13.66 |
| 16384 | **15.85** | 14.60 | 13.97 |

*(GiB total including 13.35 GiB weights, before compute buffers.)*

f16 at 16384 needs 15.85 of 15.92 GiB, and LM Studio reserves 2 GiB on top -
hence the refusal. With `-ctk q8_0 -ctv q8_0 -c 8192` it loads at **15 GiB** and
runs. Measured accuracy matched the cloud f16 run on mushoku16 to within 0.7
points (all p = 1.000).

Also llama.cpp-only: **GBNF grammars** (`grammar` body field) and
`json_schema` / `response_format` constrained decoding.

## Serving llama.cpp

```
llama-server -m MODEL.gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 -np 1 -c 8192 -ctk q8_0 -ctv q8_0 -fa on
```

State `-np 1` explicitly. LM Studio silently defaulted to `parallel 4` on the
A6000, and because it divides the context across slots that would have
quartered the effective window of a run whose entire purpose was to test a large
context.

The experiment harnesses all take `EXPERIMENT_BASE_URL`, so pointing them at a
llama.cpp server needs no code change - only `EXPERIMENT_TAG` so artifacts do
not collide.

## Packaging traps (Arch/AUR)

- `llama.cpp-hip` and `llama.cpp-vulkan` **cannot coexist**: both `Provide` and
  `Conflict` on `llama.cpp`, `libggml`, `ggml`. Installing one removes the
  other. This is packaging, not drivers - RADV and ROCm coexist fine, and both
  work on this machine simultaneously.
- To keep both binaries, build one from source into its own prefix
  (`/opt/llama-vulkan`) rather than installing the second package.
- Versions drift per variant: on one day, hip b10121, vulkan b10142, cuda
  b10154. Do not assume two machines are on the same build.

## Model management: where LM Studio actually costs time

- **`lms get` can stall silently.** On the 70B it held a live process and wrote
  **zero bytes for 113 minutes** with no error, across two attempts. Detect by
  comparing the `.part` file's mtime against the log's own start line.
- **Recovery**: HuggingFace serves range requests (`206`, `accept-ranges:
  bytes`), so `curl -C - --speed-limit 100000 --speed-time 120` resumes an
  existing partial and aborts a stall instead of hanging. Verify the partial
  starts with the `GGUF` magic first.
- **LM Studio cannot see models downloaded outside it.** Its index is
  `~/.lmstudio/.internal/gguf-metadata-cache.json`; correct folder layout is not
  enough, so `lms ls` and `lms load` both miss the file. llama.cpp reads GGUFs
  by path, or fetches them itself:

  ```
  llama-server -hf lmstudio-community/Llama-3.3-70B-Instruct-GGUF:Q4_K_M
  llama-server -cl                 # list cached models
  ```
- **LM Studio's bundled llama.cpp binaries are usable** at
  `~/.lmstudio/extensions/backends/llama.cpp-*/llama-server`, given its vendored
  libraries on `LD_LIBRARY_PATH` (`.../backends/vendor/*`). They report only
  `version: 1 (<sha>)` and lag upstream considerably - useful as a fallback when
  no system build exists.

## Building llama.cpp with CUDA on Linux

Upstream ships **no Linux CUDA prebuilt** (Windows only; Linux gets ROCm,
Vulkan, SYCL, CPU). Source build with nvcc 13 compiles every `.cu` file and then
fails at link:

```
undefined reference to `cudaGetErrorString@libcudart.so.13'
```

The libraries exist and cmake finds `libcudart.so`, but the link line carries no
`-L`. Fix:

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64"
```

## Measuring anything on the rented instance

The Thunder forwarded port costs **~0.374 s median round trip** (measured on a
bare `GET /v1/models`, no inference) against roughly **0.24 s** of actual
compute. About 60% of a short-call cloud timing is therefore network.

This invalidated an earlier reading that the A6000 was "3x slower" than the
9070 XT. On compute the two cards are comparable for this workload; **the cloud
buys VRAM capacity, not speed.** Any stack or hardware comparison must run on
the instance over loopback.

It does not follow that moving the harness to the instance is worthwhile: the
decomposition runs (~0.6 s/call, ~60% network) take minutes, while the reasoning
arms (~12.5 s/call, ~3% network) take hours and dominate the total.
