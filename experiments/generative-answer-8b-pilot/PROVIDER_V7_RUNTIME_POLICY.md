# V7 cache-bound runtime preparation policy

## Purpose and status

V7 is a new, exposed-development resource cohort for the 8B pilot. It changes one server runtime setting: `cache_ram_mib` is pinned to `0`, and the generated server command receives `--cache-ram 0`. This policy was prepared before any V7 workflow or model call. It is not evidence of a validated fix or a release qualification. Prior failures remain failures; no historical result is requalified.

The V6 provider payload is carried byte-for-byte unchanged (SHA-256 `4a782199bf3c80273dc4c8975e1020253151f6c18ee7ed4f61b69c0cbdef7570`). Model, runtime pins, prompt, generation settings, schema, validation, decoding, scoring, golds, retrieval, context, batching, and resource guard are held fixed. The 10 GiB memory cap, zero swap, four-CPU quota, 6144 context, 128 batch/ubatch and parallelism 1 remain unchanged.

## Evidence and hypothesis

The E5 job of run [40025810416](https://github.com/LuxuriantTech/evidencedesk/actions/runs/40025810416/job/120661595406) records an OOM-kill under the 10 GiB limit, a 10.0G peak, zero swap and 18 saved calls without a completed candidate result. Cache causality is unproven. The coordinator retains the original logs and verified partial artifacts. At the pinned llama.cpp revision, `--cache-ram 0` disables the shared prompt-cache pool and idle-slot snapshots, while the active slot can still reuse its common prompt prefix. This makes V7 a narrow hypothesis test. It does not establish that memory use will fall enough to avoid OOM; model residency, active KV, buffers, batch allocation, or their combination may still dominate.

## Verification boundary

`test_v7_cache_bound.py` checks manifest/config/argv agreement, the shared manifest digest, unchanged V6 provider bytes, and key unchanged generation/resource settings. These are local static/mock checks only. No model, llama-server binary, Docker, workflow, network download, inference, or quality benchmark was run as part of this preparation. A future separately frozen and reviewed cohort must retain all prior failures and report resource outcome separately from answer quality.

Primary runtime reference: [the pinned llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/9febe63d5a379f8568bae7116a5b2ba844a4a4a6/tools/server/README.md), `--cache-ram` and `--cache-prompt` options.
