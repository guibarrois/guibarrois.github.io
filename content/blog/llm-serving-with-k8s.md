+++
title = "Serving an LLM using Celery and Kubernetes"
date = "2026-08-25T00:00:00+00:00"

description = "Architecture of for a resilient and scalable system to serve a local LLM, using Celery and Kubernetes"

tags = ["ml-systems", "inference"]
+++

## Serving an LLM

With the demand for LLM increasing every month, serving an LLM is and will probably stay in the forseeable
future an important topic. In this blog post, I will present a simple architecture that uses Celery and
Kubernetes to serve a (very small) LLM in a way that is robust and, with more effort, should scale.

The service that we want to expose is very simple: a generate endpoint that take a text and complete
it by generating a certain number of tokens from a given LLM.

For instance:
```
POST /llm-api/generate
{ "text": "The capital of France is"}

{"response": "Paris.", "code": 200}
```

In the process, we will also dive a little bit about cache and memory management in k8s in order to 
understand some strange behaviors.

Let's dive in.

## Architecture

At the risk of killing the vibe, serving an LLM is not very different of serving any API: you need to
ensure that you can serve the requests from your user, with a latency that is acceptable. For an LLM
the specificity is in the fact that:
- generation requires a lot of memory to compute efficiently the forward pass,
- latency is generally in the order of magnitude of seconds.
