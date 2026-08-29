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

In the process, we will also dive in cache and memory management in k8s in order to 
understand some strange behaviors.

Let's dive in.

## Architecture

I see serving an API as optimizing around four metrics (Google's four golden signals
of monitoring):
- there is a certain load corresponding to a given number of requests by seconds (req/s),
- you want to serve all of them with an acceptable latency (measured for instance 
by the p9X latency) that is acceptable. 
- You want to do that with a number of failure (errors/s) that stays below a threshold.
- You don't won't you ressources to be saturated (%cpu or memory used).

Serving an LLM is not different, though is has:
- generation requires a lot of memory to compute efficiently the forward pass,
- acceptable latency is generally in the order of magnitude of seconds.

How does that translate in terms of architecture? In order not to deny request,
it might be a good idea to have an asynchronous system, with queuing. To be able 
to adjust the ressources, implement scaling mechanism.

Here I propose to implement that with Celery and Kubernetes in the following
architecture

[](./img/llm-serving-architecture.png)



