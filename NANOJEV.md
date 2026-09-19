# Local NanoJev decision provider

The default v2 path is fully local: Qwen remains the local worker, the
NanoJev provider is an optional CPU shadow provider, and OCR delegation is
local. The TypeSafe adapter is retained only as an optional reference.

## Pinned artifacts

- NanoJev source: `71a513bb0163b5634467842b523ee0c0ed6fb1c7`
- NanoJev checkpoint revision: `4a19595eada0857133c0d2be024f879a4077054b`
- checkpoint: `/home/brandon/models/nanojev/best.safetensors`
- checkpoint SHA-256: `fff62d1412685c1714eaa386acb603f9690371fb3cc8ad03dc41319302597c28`
- checkpoint config: `config.json` (Qwen3-0.6B backbone, 596,250,498 total
  parameters, float32 checkpoint storage)

The upstream predictor is kept in a local checkout at `vendor/NanoJev` and
is deliberately ignored by Git because it is an external source tree. The
checkout contains one small compatibility patch: CPU execution is permitted
and the CUDA-only device/automatic-cast guards are conditional. The upstream
source and checkpoint are MIT-licensed / publicly downloadable; verify the
license and revision before replacing either.

## Environment and mode

Create the project-local environment with:

```sh
python3 -m venv --system-site-packages .nanojev-venv
```

The tested host environment uses the existing system packages (Torch
2.10.0+cu128, Transformers 5.2.0, Safetensors 0.7.0, NumPy 1.26.4) and does
not change global Python or CUDA packages. NanoJev runs on CPU so Qwen's two
GPUs remain untouched. It performs one decision forward pass and returns
probability distributions; it does not autoregressively decode prose.

Run a local shadow decision:

```sh
.nanojev-venv/bin/python -m orchestrator_v2.cli run \
  --task examples/repair-task.json \
  --decision-provider nanojev --decision-mode shadow
```

`shadow` is the safe default for this provider. NanoJev judgments are
persisted as evidence and events but have no authority to mutate files,
retry operations, skip tests, change scope, or mark a task complete.

The provider defaults to `/home/brandon/models/nanojev`; set
`NANOJEV_CHECKPOINT` to another verified local directory when needed. The
source checkout can be recreated with a pinned clone of the revision above.

## Cloud Jev

TypeSafe/Jev remains optional reference code only. The normal install has no
`TYPESAFE_API_KEY` requirement and sends no project data off-machine.

## Alternative check

`mithalouni/system-one-open` was inspected as the one bounded alternative.
Its public README describes a Gemma-based open replica whose current serving
path is Modal and whose weights are on a Modal volume/Hugging Face path still
marked as pending. No verified local checkpoint was available without
retraining or a cloud/Modal deployment, so it was not added to v2.
