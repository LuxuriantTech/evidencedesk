# Current text embedding runtime

The application uses a small text-only adapter built on ONNX Runtime 1.29.0,
tokenizers 0.23.1 and NumPy 2.5.2. It loads the same pinned multilingual MiniLM
model and tokenizer files on CPU. It does not use FastEmbed's image or multimodal
integrations.

The provider still validates every listed model file by size and SHA-256 before
loading the backend. Inference uses local files only, at most 512 tokens, masked
mean pooling, 384 dimensions and L2 normalization. The provider interface, model
identity, batch size and configured answer modes are unchanged.

The current manifest is
`infra/models/paraphrase-multilingual-minilm-l12-v2-text-onnx.json`.
Its model file identities match the original manifest. The original
`infra/models/paraphrase-multilingual-minilm-l12-v2.json` remains unchanged
because frozen evaluation records bind its hash and describe the earlier
FastEmbed runtime.

## Dependencies

FastEmbed 0.8.0 and 0.9.0 require Pillow below 13 and import image-related modules
alongside the text API. Removing Pillow from that package's metadata or forcing
an incompatible version would leave an unsupported environment.

The application therefore loads the text ONNX model directly. FastEmbed and
Pillow are absent from the production dependency set. The development group
still uses Pillow 13 or later for ReportLab's synthetic PDF fixtures. A frozen
sync and the strict audit must pass without dependency overrides or ignored
advisories.

The tokenizer setup retains the applicable upstream attribution in
`third_party/fastembed.NOTICE` and `third_party/fastembed.LICENSE`.

## Validation

Run `uv sync --frozen --all-groups`, the semantic-provider and text-ONNX unit
tests, `uv run mypy`, `uv run ruff check apps/api apps/worker evals scripts`
and `uv run pip-audit --strict`. The CI also tests the full backend, web
application and real Docker Compose workflow.

A compatibility check should use the pinned model bytes and compare a synthetic
multilingual batch with the original FastEmbed 0.8.0 runtime in a separate
reference environment. Include truncation and more than 32 inputs, and deny
network connections during inference. Keep that reference environment separate
from the current application.

## Frozen evaluations

The v2-v7 configurations, recorded results and their dependency/source integrity
checks are historical evidence. They are not rewritten by this runtime change.
The legacy runtime verifier still rejects a replay when its declared FastEmbed
dependency is missing or differs. The current benchmark records the actual
text-ONNX runtime rather than claiming it used FastEmbed.

Passing runtime compatibility and software checks is not a new v7 evaluation.
The published negative v7 result remains unchanged.
