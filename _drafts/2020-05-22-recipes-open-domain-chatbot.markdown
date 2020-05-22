---
layout: post
title:  "Summary: Recipes for building an open-domain chatbot"
date:   2020-05-02 13:00:00 +0200
categories: nlp-summaries
permalink : "nlp-summaries/bert"
---
{% include scripts.html %}

> Recipes for building an open-domain chatbot ([link]())

## Why is it important?

The paper studies in depht the performance of a chatbot based in the Transformer. It shows that it's able to response in a really human way, and it's able of maintaining a chit chat conversation. However, they also show that the model lacks in-depth knowledge, it forgets facts said before, it tends to repeat what the other locutor is saying. 



Language

## What does it propose?

It constructs different chatbots based on the Transformer, and it analyses different axes of developing a chatbot. It finds that:

- Fine tuning in datasets that focus on personality, empathy, knowledge, etc. Makes the chatbot more human (even when using smaller models).
- It tries different decoding strategies, showing that beam search can be as good or better than  sampling.
- 

## Experiments

### Models

#### Retriever ([Humeau et al., 2019](https://arxiv.org/pdf/1905.01969.pdf))

<img src="/Users/constanzam/Library/Application Support/typora-user-images/Screenshot 2020-05-17 at 19.51.13.png" alt="Screenshot 2020-05-17 at 19.51.13" style="zoom:35%;" />

<u>The idea</u>: given a dialogue history (context), it retrieves the next dialogue utterance by scoring a large set of candidate responses (typically all possible training responses). 

<u>How</u>: It constructs an embedding of the context ($$y_{ctxt}$$) and one for each response candidate ($$y_{cand_i}$$), to then calculate the score of each with the dot product: $$y_{cand_i}\cdot y_{ctxt}$$.

1.  It obtains the candidates embeddings  using a transformer and an aggregator function, that can be jutake the first BERT otuput, or the average of the tokens (right side of Figure 1). 
2.  It encodes the context using other transformer and then performing an attention (left side of Figure 1), where the keys and values are the transformer output and the query is $$c_i$$ for each $$m$$ attention ($$m$$ is a hyper parameter ). It then computes another attention on tope of those embeddings, where the query is the $$y_{cand_i}$$ and the keys and values are the output from the other attention $$y^i_{ctxt}$$. In equations:

$$
\text{Transformer output} \qquad T(x) = (h_1, ..., h_N)\\
y^i_{ctxt} = \sum_jw_j^{c_i}h_j \qquad \text{where} \; (w_1^{c_i}, ..., w_N^{c_i}) = \text{softmax}(c_i\cdot h_1, ..., c_i\cdot h_N) \\
y_{ctxt} = \sum_iw_i y^i_{ctxt} \qquad \text{where} \; (w_1, ..., w_m) = \text{softmax}(y_{cand_i}\cdot y_{ctxt}^1, ..., y_{cand_i}\cdot y_{ctxt}^m)
$$

#### Generator

Standard Seq2seq model, like the transformer of "Attention is all you need" ([Aswani et. al, 2017](https://arxiv.org/abs/1706.03762), [summary here](https://cfierro94.github.io/nlp-summaries/attention-is-all-you-need)) but way bigger (90M, 2.7B, 9.4B). In comparison, Google's chatbot ([Adiwardana et. al](https://arxiv.org/abs/2001.09977)) has 2.7B parameters.

#### Retrieve and refine

Trying to solve the problems of generator models (hallucinate knowledge, unable to read and access external knowledge, dull and repetitive responses). Here they mix the two models above appending to the input of a generator model the output of a retriever model, using a special separator token. They experiment with two types:

1. **Dialogue retrieval**: it uses the dialog history and it produces a response (Same retriever architecture)
2. **Knowledge retriever**: it retrieves from a large knowledge base, where the candidates are obtained from a TF-IDF-based inverted index lookup over a Wikipedia dump. For this case a transformer is additionally trained to decide when to add the knowledge retrieval and when not to (as some context do not require knowledge).

## Training objectives 

TBD

## Decoding

They try different decoding strategies:

1. Beam search ([summary here](https://cfierro94.github.io/nlp-deep-dive/attention-is-all-you-need#beam-search))
2. Top-k sampling: at each time step the word $$i$$ is selected by sampling from the k (=10) most likely candidates from the model distribution.
3. Sample-and-rank sampling: $$N$$ independant sentences are sampled (following the model probabilities) and then the one with the highest probability is selected.

Other constraints:

1. Minimum length: Force the model to produce an answer of a defined length.
2. Predictive length: Predict (with a retriever model) the minimum length of the answer (e.g., <10, <20, <30, >30 tokens). And then we do the same as in 1.
3. Beam blocking: Force the model to not produce in the next utterance a trigram (group of 3 words) that's in the input or in the utterance itself. That can be achieved by setting to 0 the probability of the words that would create a trigram that already exists.