# Chispa: text classification from your own examples, on your own machine, with numbers you can check

*Launch post draft. Not published. Numbers and sources: [`README.md`](README.md#cifras-que-se-pueden-usar).*

There is a task almost every team that receives text keeps doing: read a short message and put a closed label on it. Is it a
complaint, a question or a request? Which team owns it? Does it need escalating? Today we are releasing **Chispa**, an open-source
(Apache-2.0) typed classification server built for exactly that.

## What it does

You send a text and one or more closed questions with their options:

```json
{"state": "I already paid, when will my delivery arrive?",
 "questions": [{"id": "q1", "question": "What does the customer need?", "kind": "choice",
                "options": [{"id": "quote"}, {"id": "order"}, {"id": "warranty"}, {"id": "advice"}, {"id": "invoice"}]}]}
```

and you get back the chosen option, a probability per option, the layer that answered and an `escalate` flag. That last one
matters: when confidence is below the threshold, Chispa says so instead of pretending, and you decide what to do with that case
(send it to a person, another model or a queue).

There are three question kinds: `choice`, yes/no, and `score` (an ordinal level; experimental).

## Three modes, depending on what you have

**You have labelled data for your task: classic mode.** You train one `.chispa` file per task with `chispa train` and serve it on
CPU. Inside it is a cascade: a very cheap linear layer answers the easy cases and, when unsure, hands over to an e5 encoder.
Trained on the massive and clinc150 train sets and measured through the real server, it reached **macro-F1 0.814 and 0.887** with
**p99 of 9 to 12 ms per query** on a cloud CPU. It ships an Android SDK and a Go mobile engine to run on the phone (mobile latency
is not measured yet, and the iOS SDK has not been compiled yet).

**You only have a few examples and no GPU: CPU mode.** No LLM. You put your examples inside the request (`examples[]`) and Chispa
answers with kNN over e5 vectors. On 4 sealed `choice` tasks, with 8 examples per class, it scored **0.483** macro accuracy —
practically the same as Qwen3-4B without examples (**0.485**) — with an encoder at **≈ 9 ms** per query on CPU.

**You have a few examples and a GPU: GPU mode.** Qwen3-4B via logprobs, with 8 demonstrations picked by kNN for each query. On the
8 sealed tasks: **0.501** overall and **0.614** on `choice`.

## Why you can trust the numbers (and their limits)

Every number in the README links its results file in the repository. The "sealed" tasks were opened exactly once, with the
configuration frozen beforehand, and results come with confidence intervals. We also publish what did not work: a list of
hypotheses that were measured and refuted (distillation, cascades without data, smaller LLMs, quantizations that lost too much).

And what we do not promise:

- Chispa does **not** classify well without examples: classic-mode zero-shot is weak.
- The GPU-mode gain over the same LLM without examples (+5.1 points overall) is **not statistically firm** with 8 tasks
  (95% CI [−4.3, +15.5]).
- `score` is experimental and only helps with a per-level rubric you write.
- The SDKs are not published to any registry yet: you use them from the repository.

## Try it in 5 minutes

You only need Go and Python; no model is downloaded:

```sh
git clone https://github.com/juan52878911/chispa && cd chispa
ejemplos/ferreteria/correr.sh
```

It trains a `.chispa` from 50 made-up examples of a hardware store's inbox (in Spanish), serves it and classifies 5 new messages.
It gets all 5 right, but all 5 come back with `escalate: true`: with 10 examples per option its confidence stays under the
threshold, and Chispa tells you. Answering on its own needs real data from your task. Expected output: [`demo.md`](demo.md).

## What is next

Confirm the GPU-mode numbers on a new sealed battery, measure latency on Android and iOS, and publish the SDKs. If you classify
short text and want to try it on your data, open an issue: we want to know where it fails.

Repository: <https://github.com/juan52878911/chispa> · Apache-2.0
