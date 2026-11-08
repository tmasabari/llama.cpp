# K2-Horizon KV streaming validation

This change wires the existing generic KV streamer into the generic attention-cache path. It selects contiguous gather mode unless the backend confirms that both paged decode and paged prefill support the model's actual K/V head dimensions and GQA ratio.

## Expected K2-Horizon selection

K2-Horizon uses 128-dimensional K/V heads and a 4:1 query-to-KV head ratio. The RDNA4 paged-attention kernel in this branch is specialized for a different shape (256 dimensions and GQA 6), so the expected RX 9070 selection is gather fallback for both decode and prefill. The specialized RDNA4 kernel is not modified.

## First validation: 16K context

Build using the existing configuration:

```bash
cd ~/code/llamaRdna
git status --short
cmake --build build-rx9070 --parallel 8
```

Run the K2-Horizon model at 16K:

```bash
MP_PLACES=cores OMP_WAIT_POLICY=active taskset -c 0-7 \
./build-rx9070/bin/llama-server \
  --model /mnt/8462F59B62F591E2/models/K2/K2-Horizon-4B-Q6_K.gguf \
  --alias k2horizon4b \
  --host 127.0.0.1 --port 8080 \
  --device ROCm0 --gpu-layers auto \
  --ctx-size 16384 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn on -fit off \
  --parallel 1 --kv-unified --kv-unified-per-slot 16384 \
  --kv-stream --kv-resident-mib 450 \
  --jinja --reasoning-preserve \
  --load-mode mmap+mlock \
  --log-verbosity 4
```

Confirm the startup log reports paged decode and prefill as gather fallback, and that initialization no longer throws `KV streaming is not supported by this model's memory`.

Then compare the same deterministic prompt against the non-streaming 16K baseline. Record generated text, token count, prompt/decode throughput, peak VRAM, and host RAM. Successful startup alone is not sufficient evidence of correctness.

## Required test ladder

1. 16K: initialization, prompt evaluation, and at least 256 generated tokens.
2. 32K: repeat with `--kv-unified-per-slot 32768`.
3. Long decode: generate at least 2,000 tokens after a substantial prompt so page eviction and host-to-device copies occur.
4. Repeat with a second prompt that causes a different set of pages to be streamed.
5. Compare against non-streaming output with fixed seed and deterministic sampling. Investigate semantic divergence, NaNs, stalls, or missing tokens.

Do not increase context or reduce the resident budget until the 16K gather path passes. Do not enable the decode resident budget for the first run.

## Validation status and limitations

- This patch has not been compiled or run on the RX 9070 by the patch author.
- It does not add a 128-dimensional RDNA4 paged-attention kernel.
- It does not prove that the model's full 512K metadata context fits in 16GB VRAM; streaming only changes KV residency, not model weights, compute buffers, or other allocations.
