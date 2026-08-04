+++
title = "KV Cache"
date = "2026-07-15T22:22:23+02:00"

description = "A from-scratch walkthrough of the KV cache: counting the computations of attention and seeing where caching turns a quadratic cost into a linear one."

tags = ["ml-systems", "inference"]
+++

# KV cache

Implementing KV cache is the standard exercise for everyone trying to get a deeper understanding of transformer 
architecture, so I decided to do a deep dive on it to be sure I understand it well.

We are going to go through the computations taking place in the attention layer and then observe how the 
addition of KV cache makes the quadratic term disappear. We will also compare it to results obtained with a
simple KV cache implementation.

Let's dive in !

## Computation cost of attention

Let's consider that each word is a token, and that our model is a next token predictor. We have processed so
far:

```
And I think to myself what a wonderful
```

And we want to predict the next token.

The context will therefore be `to myself what a wonderful`. The first step of the attention layer 
consists in retrieving the embedding of each token. It is a simple lookup table, so no computation here.
We obtain 5 vectors of `emb_size` length, let's say 128 in our case.

In the transformer architecture, you notoriously then have to compute for each forward pass three matrices,
K (key), Q (query) and V (value) that are then combined.

K, Q, V matrices are obtained by applying a linear layer to the embedding and projecting it
onto another vector of size `emb_size`. So K, Q and V are here of size `(5, 128)`.

How much computation is that? A linear layer does `K = x @ Aᵀ + b`. In our case
- x is `(seq_length, emb_size) = (5, 128)`
- A is `(emb_size, emb_size) = (128, 128)`
- b is `(1, emb_size) = (1, 128)`

The number of computations is therefore three times (for K, V and Q):
`2 * seq_length * emb_size * emb_size + seq_length * emb_size = 2 * 5 * 128 * 128 + 5 * 128`

After that, we compute an intermediary matrix that represents the weights (W): how much token\[i\] attends
to token\[j\]:

```
W = Q @ Kᵀ / √emb_size # shape = (5, 128) @ (128, 5) -> (5, 5).
W = W.masked_fill(upper_triangular, -inf) # causal mask: tokens can only attend to previous tokens (see footnotes)
W = softmax(W) # this is to normalize the weights
```

(Note that in the case of multihead attention, the embedding dimension is divided by the number of heads and 
the result for W is `(n_head, 5, 5)`, but that doesn't change the mechanism).

Why call them query and key? Conceptually, Q_i represents what the token at position i is
looking for (query), while K_i represents what token `i` advertises about itself (key).

But how is it translated in this computation? My mental model is that with the matrix multiplication, 
the W matrix is going to have at row `i` the result of the query of the token `i` with all the keys in
the context, therefore Q is what plays the role of query at token `i`. 

How much does that cost? Matrix multiplication is `2 * seq_length * seq_length * emb_size = 2 * 5 * 5 * 128`
computations, softmax is about `3 * seq_length * seq_length = 3 * 5 * 5`

Almost there, once we have the matrix W, we retrieve the value associated with the weights, that is
also a matrix multiplication

```
out = W @ V # shape = (5, 5) @ (5, 128) -> (5, 128)
```

After that there is one last linear layer, called projection. Its role is to project back into the
residual stream. I think of it as follows: attention has worked by projecting vectors in whatever space is
the most useful to do its attention job. Now we want to project it back into the output space.

```
out_proj = out @ B + b
```

Again that's `2 * seq_length * emb_size * emb_size + seq_length * emb_size = 2 * 5 * 128 * 128 + 5 * 128`

Let's put together all the computation costs:

K, V, Q -> `6 * seq_length * emb_size^2 + 3 * seq_length * emb_size`
W and softmax -> `2 * seq_length^2 * emb_size + 3 * seq_length^2 = seq_length^2 * (2 * emb_size + 3)`
out -> `2 * seq_length^2 * emb_size`
Projection -> `2 * seq_length * emb_size^2 + seq_length * emb_size`

This is of the form: `seq_length^2*a + seq_length*b (with a and b different constants from the formula above)`

## KV cache

If you go back to the previous computation, you realise that when the context grows, 
a lot of the computations are repeated:
- the K, V and Q matrices have rows that correspond to each token, so when the context grows,
  only the last row (last token) in the context is new. All the other rows are
  identical.
- the W matrix has only one new row, which corresponds to the last row in Q multiplied by K.
The principle of KV cache is, well, to cache those computations. This is not free of course: you trade FLOPs for Memory but we will come back to this tradeoff
later.

To go back to our previous computations, if n is the index of the last token, that means that 
you can compute only K_n, Q_n and V_n, the computation is therefore
`2 * 1 * emb_size * emb_size + 1 * emb_size = 2 * 1 * 128 * 128 + 128`
But the cache mostly pays off in the computation of W, where you can just compute the last
row:

```
W_n = Q_n @ Kᵀ / √emb_size # shape = (1, 128) @ (128, 5) -> (1, 5).
out_n = W_n @ V # shape = (1, 5) @ (5, 128) -> (1, 128)
out_proj_n = out_n @ B + b # shape = (1, 128) @ (128, 128) -> (1, 128)
```

which costs this time respectively: 
```
2 * 1 * seq_length * emb_size = 2 * 1 * 5 * 128
2 * 1 * seq_length * emb_size = 2 * 1 * 5 * 128
2 * emb_size^2 + emb_size
```
And here, you can see that the factor that was ~seq_length^2 becomes ~seq_length

Therefore, the final form of the computation is something of `seq_length*c + d`! Now we can compare
the computation costs with and without KV cache, step by step:

| computation   | without cache                                             | with cache                                   |
|---------------|-----------------------------------------------------------|----------------------------------------------|
| K, Q, V       | `6 * seq_length * emb_size^2 + 3 * seq_length * emb_size` | `6 * emb_size^2 + 3 * emb_size`              |
| W and softmax | `2 * seq_length^2 * emb_size + 3 * seq_length^2`          | `2 * seq_length * emb_size + 3 * seq_length` |
| out           | `2 * seq_length^2 * emb_size`                             | `2 * seq_length * emb_size`                  |
| projection    | `2 * seq_length * emb_size^2 + seq_length * emb_size`     | `2 * emb_size^2 + emb_size`                  |

Each cell of the "without cache" column is exactly `seq_length` times the corresponding
cell of the "with cache" column. We could have seen that coming: intuitively, a full forward pass 
processes each of the `seq_length` tokens as the newest one, while the version with KV cache
only does it for one.

We should therefore end up with a ratio between the two versions that increases linearly with seq length: 

| seq length                             | 16 | 128 | 256 | 512 | 1024 | 2048 |
|----------------------------------------|----|-----|-----|-----|------|------|
| FLOPs(without cache)/FLOPs(with cache) | 16 | 128 | 256 | 512 | 1024 | 2048 |

## Experiments

To reproduce those results, I implemented a toy model with and without KV cache. With this model, we can:
1. Evaluate the flops using the handy `FlopCounterMode` of torch
2. Evaluate the latency

Let's look at the FLOPs for each seq length:

![FLOPs to predict one token, with and without KV cache](/images/kv-cache-flops.png)

This is exactly what the theory predicted! The FLOPs of the cached version increase linearly 
with the context size while the version without cache increases with its square. (Both curves only
reach their asymptotic slopes once seq_length outgrows the constant per-token work, that is the
flat start of the cached curve.) The ratio of the two curves gives us exactly the table above,
down to the decimal: 16.00, 128.00, ..., 2048.00.

Let's now look at the latency:

![Time to predict one token, with and without KV cache](/images/kv-cache-latency.png)

Here, the picture is more blurry: the improvement is real, but it goes from ~6x at short context to
~38x at 1024 tokens — far from the 1024x that the FLOP count promises. That should not surprise us:
FLOPs are not the only thing that takes time. Latency is also built by fixed costs that do not shrink
with the FLOPs (kernel launches, memory access, Python dispatch). The version without cache at least
keeps the CPU busy doing actual arithmetic.


## Memory vs FLOP tradeoff

But what did we spend in memory to save this amount of FLOPs? K and V are both of size `seq_length * emb_size`, 
so for 32 bits floats the cache takes `4 * 2 * seq_length * emb_size` bytes:

| seq length (emb_size = 128)                   | 16    | 128  | 256  | 512  | 1024 | 2048 |
|---------------------------------------------- |-------|------|------|------|------|------|
| Supplementary memory spent on cache (in MB)   | 0.016 | 0.13 | 0.26 | 0.52 | 1.05 | 2.10 |

That is not a negligible amount, in particular when seq length grows and when there are multiple
attention layers, since each layer keeps its own K and V.

In the real world, the number of layers goes from ~10 to ~100, but there is also half precision
(32 bits floats -> 16 bits floats) and GQA. GQA (grouped-query attention) consists in sharing one
K/V pair between a group of query heads (typically 4 to 8) inside each layer, while still computing
a different Q per head. That divides the width of the cached K and V and the memory footprint
by as much.

Putting it together, the cache costs `2 (K and V) * n_layers * kv_width * 2 bytes` per token:

| Model        | Layers | Embedding size | GQA      | KV width | Memory per token (fp16) | Max context | Cache at max context |
|--------------|--------|----------------|----------|----------|-------------------------|-------------|----------------------|
| GPT-2 small  | 12     | 768            | No       | 768      | 0.04 MB                 | 1k          | 38 MB                |
| Llama-3-8B   | 32     | 4096           | Yes (4x) | 1024     | 0.13 MB                 | 128k        | 17 GB                |
| Llama-3-70B  | 80     | 8192           | Yes (8x) | 1024     | 0.33 MB                 | 128k        | 43 GB                |

At full context, that is ~43 GB of cache for a single sequence on the 70B model. This is why the KV cache, not compute, 
is often what limits context length and batch size in real deployments.


## Footnotes

### Handling of the position embedding

In "classic" attention, the input to the attention layer is the embedding of the token plus the 
positional embedding. That means that the KV cache is usable only when the context window is 
growing. If the context window is just moving to the right, the position of each token 
changes, and the cache is invalidated.

In modern positional embeddings this constraint disappears: for instance with RoPE,
keys are still position-dependent, but attention scores depend only on relative positions, 
so tokens never need to be renumbered: a sliding window just evicts old cache entries and keeps the rest.

### Causal masking

The KV cache only works here because in autoregressive models, tokens can only attend to previous
tokens. That is why we apply a causal mask on the W matrix in the attention computation.
Without that we would have to compute, for each new token, the attention of the previous tokens to
the new token, and could not use the cached version of the rows.

That means that the KV cache as explained here only applies to this kind of model, but not
in the general case of attention. For instance in the BERT architecture, attention is not
causal, therefore the KV cache can not be applied.


Text were written by hand, and proofread by claude. Scripts have been written by me. Schema
were generated by Claude.
