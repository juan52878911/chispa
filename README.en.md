<p align="center">
  <img src="docs/marca/chispa.svg" width="128" alt="Chispa icon: a spark jumping from three slivers of kindling">
</p>

<h1 align="center">Chispa</h1>

<p align="center">
  <strong>Open-source typed classification server: one text, one closed question (choice, yes/no or score), one answer with a probability per option.</strong>
</p>

<p align="center"><a href="README.md">Español</a> (full documentation, in Spanish) · <a href="LICENSE">Apache-2.0</a></p>

---

## Who it is for

Anyone who has to put a closed label on many short texts (support tickets, citizen requests, emails, chat messages) and wants the
decision to run on their own machine, from their own examples, with a probability per option and an explicit "not sure" signal
(`escalate`) instead of a made-up answer.

| If you have… | Use | What is measured |
|---|---|---|
| Labelled data for your task (≥ 20 per option) and CPU only, or a phone | **Classic mode**: a `.chispa` model trained per task | macro-F1 **0.814** on massive and **0.887** on clinc150, p99 **9–12 ms** per query on CPU ([source](server/deploy/resultados/e2e-20261028/LEEME.md)) |
| A few examples per class (8 in the measurement), CPU only | **CPU mode**: kNN over e5 embeddings, examples sent in the request | `choice` **0.483** on 8 sealed tasks, on par with Qwen3-4B without examples (0.485); encoder ≈ 9 ms per query ([numbers](README.md#cifras-honestas), [`PRODUCTO.md`](server/PRODUCTO.md)) |
| A few examples per class and a GPU | **GPU mode**: Qwen3-4B with 8 demonstrations picked by kNN | **0.501** overall on the sealed tasks (0.614 on `choice`) ([numbers](README.md#cifras-honestas)) |

What it is not: it does not classify well without examples (classic-mode zero-shot is weak), `score` is experimental and only helps
with a rubric you write, the GPU-mode gain over the same LLM without examples (+5.1 overall) is not statistically firm with 8 tasks,
and no SDK package is published to a registry yet.

## Try it in 5 minutes

CPU only, no model downloads; needs Go ≥ 1.27 and Python ≥ 3.9. From the repository root:

```sh
ejemplos/ferreteria/correr.sh
```

It trains a `.chispa` from 50 made-up examples (10 per option) of a hardware store's inbox, serves it with `chispa-server` and
classifies 5 unseen messages. With so few examples it gets all 5 right but **escalates** all 5, because its confidence stays under
the threshold: that is the intended behaviour, and answering on its own needs more data. Expected output: [`ejemplos/README.md`](ejemplos/README.md).

## Where to go next

- Two-mode server (CPU/GPU), configs, request examples and limits: [`server/PRODUCTO.md`](server/PRODUCTO.md); API contract: [`server/api/openapi.yaml`](server/api/openapi.yaml).
- Every number, its confidence interval and what was refuted: [README (Spanish), «Cifras honestas»](README.md#cifras-honestas).
- Classic mode docs: [`docs/`](docs/00-indice.md). SDKs (Python, JS, Go, Android, Go mobile engine; iOS not compiled yet): [`sdk/`](sdk/README.md).
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Security: [`SECURITY.md`](SECURITY.md) · Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

Apache-2.0 for code and weights ([`LICENSE`](LICENSE)); third-party attributions in [`NOTICE`](NOTICE).
