+++
title = "Memory usage of an LLM served via Celery and Kubernetes"
date = "2026-08-25T00:00:00+00:00"

description = "Architecture of for a resilient and scalable system to serve a local LLM, using Celery and Kubernetes"

tags = ["ml-systems inference"]
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


## Experimentation

In order to experiment with the limits and ressources settings in
kubernetes, I conducted a small experiments to find the minimal memory 
that needs to be set to run the worker.

My mental model was something like
- the weights of the model are approximately 550Mb, so the minimum to
initialize the pod should be around 600Mb.
- inference uses additional memory for the kv cache, so the limit to
do a generation should be a little bit higher.

| Memory limit | Initialization | Decoding speed |
|:-------------|:---------------|:---------------|
| 100M         | OOM            | —              |
| 200M         | OOM            | —              |
| 300M         | OOM            | —              |
| 350M         | OOM            | —              |	
| *400M*         | *OK*             | *5.2 token/s*    |
| 500M         | OK             | 12.5 token/s   |
| 1G           | OK             | 14.3 token/s   |
| 2G           | OK             | 18.2 token/s   |

Everything behave more or less as expected until 400M: the worker
tries to load the model in memory, it is too big, therefore
it is killed. But at 400M, the worker is initialized, the model
is loaded, and you can even run inference (though very slowly).

What is going on here ? The model is 124M parameters, stored in 32 float. 
This is 3,968bits x 124M = 496Mb, so how can it fit the 400Mb limit ?

## How RAM is handled in the kubernetes worker

There are several hypothesis that could explain this strange fact:

_Is the size of the model overestimated ? No_

First thing first, let's try to mesure the memory taken by the parameters, see
if this is in line with the estimation of 496Mb above. We can do that
by running this simple python command after the loading of the model:
```
parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in _model.parameters()
        )
```
...and we arrive at exactly at 498Mb.

_Does the pod have more memory available than its limits ? No_

In kubernetes on linux, the handling of the memory limit is handled to 
a linux mechanism, named `cgroup`. A cgroup is an ensemble of processes 
with kernel enforced boundaries, among which a max memory. It is
possible to inspect the memory usage and limit of those cgroups

```
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
```
On our cgroup of interest (the one corresponding to our "magical pod"),
this return respecively

```
memory.max=399998976 bits
memory.current=395124736 bits
```

Well, the memory is full, but the 400M limits is respected. But
how is it possible that the memory used stays below the size of the
model, while the worker is still able to serve it ?

Well, turns out this is possible thanks to another mechanims: `mmap`

### `mmap` or fake it until you access it

`mmap` is a way to handle memory that apply a lazy loading technique to
memory handling: the idea is that the data that needs to be in RAM is 
split in pages, but pages are loaded in RAM when the process actually
accesses it.	

Let say that you have data that can be split in four pages, but only
two are used by the process. In practice, only those pages need to
be resident in RAM:
```
data
[ A ][ B ][ C ][ D ]

RAM
[ A ][ B ]
```
When page D is accessed, this causes a page fault, Linux makes it
available in RAM and the process continues:

```
data
[ A ][ B ][ C ][ D ]

RAM
[ A ][ B ][ D ]
```

But what happens when there is not enough RAM to handle the whole data ?
The Linux kernel can reclaim file-backed pages: it removes somes pages that
can be loaded again from later from the original file.

```
RAM
[ A ][ B ][ D ]
        ↓
page C is needed
        ↓
Linux kernel reclaims page B
        ↓
RAM
[ A ][   ][ D ]
        ↓
page C is faulted in
        ↓
RAM
[ A ][ C ][ D ]
```

So maybe this is what is going on whith our model: it is actually not
fully loaded into RAM, and during for instance inference, there
is a succession of reclaiming / faulting, for instance to load
successively the weights of the layers.

That would also explain the increase in latency observed when the 
memory limit on the pod decreases: because memory is smaller, there are more
reclaiming / faulting cycles to do, leading to more disk access, and 
more latency.

To confirm that it is what is going on, we can analyze of the memory
is distributed anonymous (non file-backed,therefore non reclaimable) and file-backed
(memory):

```
cat /sys/fs/cgroup/memory.stat
```
returns
```
anon 375549952
file 14557184
```

This was not what I was expected... Only around 15MB of the RAM is file-backed,
the rest is anonymous. For my explanation about mmap to be true,
that would mean that the 498MB of the weights are distributed:

```
anon memory + file-backed memory + file-backed not in memory = 498MB
with file-backed memory ~= 15MB
and  file-backed memory ~= 108Mb
```

In the specific case of the 400MB memory limit, the file-backed memory is
around 15MB. Sothat means that the "sliding memory subset" that allows to read 
successively the rest of the weights that are not in memory is quite small.
Not impossible, but less than expected !

## Conclusion

It would probably be possible to continue the investigation to rule out 
or confirm this hypothesis, but at this point, I feel I have already learned 
a lot about memory limit and memory handling in cgroup, so I leave the 
rest of the investigation to a motivated reader.

If someone has an alternative explanation for the 400MB worker serving
a 498MB model, I would also be very glad to hear it !
