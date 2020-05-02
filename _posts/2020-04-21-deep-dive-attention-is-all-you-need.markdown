---
layout: post
title:  "Deep dive: Attention is all you need."
date:   2020-04-25 18:00:00 +0200
categories: nlp-deep-dive
permalink : "nlp-deep-dive/attention-is-all-you-need"
mathjax : true

---

{% include scripts.html %}

> The objective of this article is to understand the concepts on which the transformer architecture ([Vaswani et. al](https://arxiv.org/abs/1706.03762)) is based on.<br>
> If you want a general overview of the paper you can check the [summary](https://cfierro94.github.io/nlp-summaries/attention-is-all-you-need).

Here I'm going to present a summary of:

- [Byte pair encoding](https://cfierro94.github.io/nlp-deep-dive/attention-is-all-you-need#byte-pair-encoding)
- [Beam search](https://cfierro94.github.io/nlp-deep-dive/attention-is-all-you-need#beam-search)
- [Label smoothing](https://cfierro94.github.io/nlp-deep-dive/attention-is-all-you-need#label-smoothing)
- [Dropout](https://cfierro94.github.io/nlp-deep-dive/attention-is-all-you-need#dropout)
- [Layer Normalization](https://cfierro94.github.io/nlp-deep-dive/attention-is-all-you-need#layer-normalization)

## Byte Pair Encoding

#### Context

This is an algorithm to define the tokens for which we're going to learn vector embeddings. The simplest way to do this is consider each word and punctuation in the text as a token. The problem with this approach is that in testing we won't have an embedding for words we didn't see before. Some research has successfully used characters as tokens ([Kim et. al 2016](https://arxiv.org/abs/1508.06615), [Costa-Jussa et. al 2016](https://arxiv.org/abs/1603.00810)). Byte pair encoding can be put in the middle of these two techniques.

#### The algorithm

The motivation behind the algorithm is to define a set of tokens that can be used to construct any word, but also contain the most typical words. So we can learn good representations for the most common terms but at the same time remain flexible and have some knowledge for unknown words. The algorithm then is as follows:

1. We start the tokens set with each of the possible characters plus an end-word character.
2. We determine the number of merges that we want to do.
3. For every merge, we will count the occurences of each pair of tokens in our corpus, and we're going to add as a string the pair (the concatenation) the most frequent. Therefore, adding 1 new token to our set.

With this, the size of the vocabulary = the number of merges + the number of different characters + 1 (the end-word character). So if we define the number of merges as $$\inf$$ then the vocabulary is all the possible characters and all the different words.

## Beam Search

#### Context

There are different algorithms to decode the final output sequence. This is because our model outputs a probability distribution over our vocabulary, and then we need to choose one word each time until we arrived at the end character. One option (*greddy decoding*) would be to choose at each step the word with the highest probability, but the problem is that this may not lead to the highest probability sentence because it's calculated as:

$$P(w_0...w_n) = \prod_i P(w_i) \\
\iff \log(P(w_0...w_n)) = \sum_i \log(w_i)
$$

#### The algorithm

Instead of being that greedy, beam search proposes to maintain `beam_size` hypothesis (possible sentences). Then at each step:

  1. We predict the next `beam_size` tokens for each hypothesis
  2. From all those possible sentences we take the `beam_size` most probable hypothesis.

 We can stop when we complete $$n$$ sentences (arrived at the end character), or after $$t$$ steps. Additionally, they propose to normalize (to divide) the sentence probability by $$\alpha$$, so longer sentences are not less probable.

## Label smoothing

[(Szegedy et. al)](https://arxiv.org/abs/1512.00567)

This is a regularization technique that encourages the model to be less confident, therefore more adaptable.

In classifications problems we have ground truth data that follows a distribution $$q$$ that we usually define as a one hot vector:

$$
q(y|x_i) = \delta_{y,y_i} := \left\{\begin{array}{ll}1 & \text{if} \quad y=y_i \\0 & \text{if} \quad \text{otherwise} \\\end{array} \right.
$$

If we use the softmax to calculate the output probabilities and we use cross entropy as the loss, then this labels representation can lead to overfitting, therefore this technique proposes to use smoother labels.

### Understanding the problem

In classification problems where we predict the label with a softmax as

$$p(y_j|z) = \frac{\exp(z_j)}{\sum_i \exp(z_i)}$$

Where $$x$$ is the input, $$y_j$$ is one of the possible labels, and $$z$$ the logits (the output score of our model). And we use the cross entropy loss as shown below for one example ($$l$$) and for all of them ($$L$$):

$$
l = H(q,p) = - \sum_{y=1}^K q(y|x) \log p(y|x)\\\implies L = - \sum_{i=1}^n \sum_{y=1}^K q(y|x_i) \log p(y|x_i)
$$

When we take the ground truth distribution discrete, that is $$\delta_{y,y_i}$$ (see definition above of $$q$$), then $$q=0$$ for all $$y$$ different than $$y_i$$ (the correct label for element $$i$$), then:

$$
l = - \log p(y_i|x_i)\\
$$

Let's now calculate the derivative of this to find the minimum of the loss,

$$
\frac{\partial l}{\partial z_k} = \frac{\partial}{\partial z_k}\bigg(-\log \Big(\frac{\exp(z_i)}{\sum_j \exp(z_j)}\Big) \bigg)\\\iff \frac{\partial}{\partial z_k} \bigg( \log \Big(\sum_j \exp(z_j)\Big) - z_i \bigg)\\\iff \frac{1}{\sum_j \exp(z_j)}\frac{\partial}{\partial z_k}\Big(\sum_j \exp(z_j)\Big) - \frac{\partial z_i}{\partial z_k}\\\iff \frac{\exp(z_k)}{\sum_j \exp(z_j)} - \delta_{z_i=z_k}\\\iff p(y_k) - q(y_k)
$$

Thus,

$$
\frac{\partial l}{\partial z_k} = p(y_k) - q(y_k) \in [-1, 1]\\
$$

Then the function is minimized (it's derivative is zero) when $$p(y_k) = q(y_k)$$. Which is approachable if $$z_i >> z_k \; \forall i \ne k$$, in words having the correct logit way bigger than the rest. Because with these values the softmax $$p$$ would output 1 for the $$i$$ index and zero elsewhere. This can cause two problems:

1.  If the model learns to output $$z_i >> z_k$$, then it will be overfitting the groundtruth data and it's not guaranteed to generalize.
2. It encourages the differences between the largest logit and all others to become large, and this, combined with the fact that the gradient is between -1 and 1 , reduces the ability of the model to adapt. In other words, the model becomes too confident of its predictions.

### Proposed solution

Instead of using a one hot vector, we introduce a noise distribution $$u(y)$$ on the following way:

$$
q'(y|x_i) = (1-\epsilon)\delta_{y,y_i} + \epsilon u(y)
$$

Thus we have a mixture between the old distribution $$q$$ and the fixed distribution $$u$$, with weights $$(1-\epsilon)$$ and $$\epsilon$$. We can see this as in, for a $$y_j$$ label we first set it to the ground truth label  $$\delta_{y_j,y_i}$$ and then with probability $$\epsilon$$ we replace the label with the distribution $$u$$.

In the [paper](https://arxiv.org/pdf/1512.00567) where this regularization was proposed they used the uniform distribution $$u(y) = 1/K$$. If we look at the cross entropy now, it would be:

$$
H(q', p) = - \sum_{y=1}^K q'(y|x) \log p(y|x)\\= (1-\epsilon)H(q,p) + \epsilon H(u, p)
$$

Where the term $$\epsilon H(u, p)$$ is penalising the deviation of $$p$$ from the prior $$u$$, because if these two are too alike, then its cross entropy ($$H(u, p)$$) will be bigger and therefore the loss will be bigger.

## Dropout

[(Srivastava et. al)](http://jmlr.org/papers/v15/srivastava14a.html)

This is another regularization technique that is pretty simple but highly effective. It turns off neurons with a probability $$(1-p)$$, or in other words it keeps neurons with a probability $$p$$. Doing this the model can learn more relevant patterns and is less prone to overfit, therefore it can achieve better performance.

The intuition behind dropout is that when we delete random neurons, we're potentially training exponential sub neural networks at the same time! And then at prediction time, we will be averaging each of those predictions.

In test time we don't drop (turn off) neurons, and since it is not feasible to explicitly average the predictions from exponentially many thinned models, we approximate this by multiplying by $$p$$ the output of the hidden unit. So in this way the expected output of a hidden unit is the same during training and testing.

## Layer Normalization

[(Lei Ba et. al 2016)](https://arxiv.org/pdf/1607.06450.pdf)

Motivation: During training neural networks converge faster if the input is whitened, that is, linearly transformed to have zero mean and unit variance and decorrelated ([LeCun et. al 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)). So we can see the output of one layer as the input of another network, therefore it's clear that normalizing the intermediate values in the network could be beneficial.  The main problem is that each layer has to readjust to the changes in the distribution of the input.

This problem was presented by [Ioffe et. al 2015](https://arxiv.org/pdf/1502.03167), where they proposed Batch Normalization to overcome this issue, as a way to normalize the inputs of each layer in the network.

### Batch Normalization

In a nutshell, we can think of Batch Normalization as an extra layer after each hidden layer, that transforms the inputs for the next layer from $$x$$ to $$y$$. If we consider $$\mathcal{B} = \{x_1...x_m\}$$ to be the mini batch where each $$x_j = (x_j^{(1)}...x_j^{(H)})$$ is an input vector of a hidden layer, then the normalization of each dimension is the following:

$$
\hat{x}^{(k)} = \frac{x^{(k)} - \text{E}[x^{(k)}]}{\sqrt{\text{Var}(x^{(k)})}}
$$

We'll approximate the expectation ($\mu$) and the variance ($$\sigma^2$$) calculating them at the mini batch level. Then the batch normalization will be:

$$
\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m x_i\\\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_\mathcal{B})\\\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}\\y_i = \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma,\beta} (x_i)
$$

Where $$\epsilon$$ is a constant added for numerical stability, and $$\gamma$$ and $$\beta$$ are parameters of this "layer", learnt through backpropagation. These parameters are used to be able to keep the representational power of the layer, so by setting $$\gamma = \sqrt{\sigma^2_\mathcal{B}}$$ and $$\beta = \mu$$ we can recover the original output, if it were to be the optimal one.

Additionally, during inference we'll use $$\gamma$$ and $$\beta$$ fixed and the expectation and variance will be computed over the entire population (using the first equation).

### Back to Layer Normalization

Batch normalization is not easily extendable to Recurrent Neural Networks (RNN), because it requires running averages of the summed input statistics, to compute $$\mu_\mathcal{B}$$ and $$\sigma^2_\mathcal{B}$$. However, the summed inputs in a RNN often vary with the length of the sequence, so applying batch normalization to RNNs appears to require different statistics for different time-steps. Moreover it cannot be applied to online learning tasks (with batch size = 1).

So they proposed layer normalization, that normalises the layers as follows:

Let $$a^l$$ be the vector of outputs of the $$l^{\text{th}}$$ hidden layer, and $$a^l \in R^H$$ (each hidden layer has $$H$$ hidden units), then:

$$
\mu^l = \frac{1}{H} \sum_{i=1}^H a_i^l \qquad \sigma^l = \sqrt{\frac{1}{H}\sum_{i=1}^H (a_i^l - \mu^l)^2}
$$

This looks really similar to the above equations for $$\mu_\mathcal{B}$$ and $$\sigma^2_\mathcal{B}$$, however the equations here use only the hidden layer output whereas the ones above use the whole batch.

Similarly as BN, we'll learn a linear function ($$\gamma$$ and $$\beta$$) or as they call it in the paper, a gain function $$g$$.

Unlike BN, LN is used the same way in training and test times.

#### Comparison to BN

In the paper they showed that LN works better (converges faster and it's robust to changes in the batch size) for RNNs and feed-forward networks. However BN outperforms LN when applied to CNNs.
