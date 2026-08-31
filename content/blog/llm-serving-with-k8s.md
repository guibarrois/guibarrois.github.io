+++
title = "Serving an LLM using Celery and Kubernetes"
date = "2026-08-25T00:00:00+00:00"

description = "Architecture of for a resilient and scalable system to serve a local LLM, using Celery and Kubernetes"

tags = ["ml-systems inference"]` for the worker

A redis image, that serve both as the broker and the backend is also built.
+++

## Serving an LLM

With the demand for LLM increasing every month, serving an LLM is and will probably stay in the forseeable
future an important topic. In this blog post, I will present quickly a simple architecture that uses Celery and
Kubernetes to serve a (very small) LLM in a way that is robust and, with more effort, should scale.

The service that we want to expose is very simple: a generate endpoint that take a text and complete
it by generating a certain number of tokens from a given LLM.

For instance:
```
POST /llm-api/generate
{ "text": "The capital of France is"}

{"response": "Paris. code": 200}` for the worker

A redis image, that serve both as the broker and the backend is also built.
```

After that I will go deep into why, even with a memory limits in our kubernetes pods that was below
the size of the model, pods was not killed and we were still able to serve it. That will lead us 
into a better understanding of memory management when serving an LLM.


Let's dive in.

## Architecture

Serving an API is optimizing around four metrics (Google's four golden signals
of monitoring):
- there is a certain load corresponding to a given number of requests by seconds (req/s),
- you want to serve all of them with an acceptable latency (measured for instance 
by the p9X latency) that is acceptable. 
- You want to do that with a number of failure (errors/s) that stays below a threshold.
- You don't won't you ressources to be saturated (%cpu or memory used).

Serving an LLM is not different, though is has some specificities:
- generation requires a lot of memory to compute efficiently the forward pass,
- acceptable latency is generally in the order of magnitude of seconds.

How does that translate in terms of architecture? In order not to deny request,
it might be a good idea to have an asynchronous system, with queuing. To be able 
to adjust the ressources, implement scaling mechanism.



## Implementation

### Architecture

[](./img/llm-serving-architecture.png)

You can find the code [here](https://github.com/guibarrois/gpt2_constrained).
The implementation is fairly mininmal:

The API contains two endpoints
- a POST /generate endpoint that triggers the job and return its id
- a GET /result/<job_id> endpoint that collects the result

The Celery worker has a generate task that complete
a text (passed as parameter) with a certain number of tokens (also passed
as parameter). This generate task is triggered by the generate endpoint, that 
returns the id assigned by the worker.

API and worker share the same docker image, that is called with:
- `gunicorn --bind 0.0.0.0:5000 app:app` to serve the API
- `celery -A tasks worker --concurrency=1 --loglevel=info` for the worker

A redis image, that serve both as the broker and the backend is also built.

### Pre loading of the weights

In order not to reload the weights of the model at each calls to generate,
model loading is done in a separate task with the `@worker_process_init` 
decorator. The model is cached on a Persistent Volume, attached to the
k8s Pod with a Persistent Volume Claim.

### Deployement

This is all deployed with k8s



