---
layout: post
title:  "Summary: Recipes for building an open-domain chatbot"
date:   2020-05-23 20:00:00 +0200
categories: nlp-summaries
permalink : "nlp-summaries/recipes-open-domain-chatbot"
---
{% include scripts.html %}

> ([Roller et. al, 2020](https://arxiv.org/abs/2004.13637)). This paper studies in depth the performance of a chatbot based on the Transformer. It shows that it's able to respond in a really human way, and it's able to maintain a chit chat conversation. However, they also show that the model lacks in-depth knowledge, it forgets facts stated before and it tends to repeat what the other locutor is saying.

### What does it propose?

It constructs different chatbots based on the Transformer, and it analyses different axes of developing a chatbot. It finds that:

- Fine tuning on datasets that focus on personality, empathy, knowledge, etc. Makes the chatbot more human (even when using smaller models).
- It tries different decoding strategies, showing that beam search can be as good or better than  sampling.
- It presents the flaws of the developed models.

## Models

### Retriever 

([Humeau et al., 2019](https://arxiv.org/pdf/1905.01969))

{% include image.html file="../assets/img/nlp-summary-03/poly_encoder.png"
description="Figure 1. Poly encoder architecture." zoom=45 %}

<u>The idea</u>: given a dialogue history (context), it retrieves the next dialogue utterance by scoring a large set of candidate responses (typically all possible training responses). 

<u>How</u>: It constructs an embedding of the context ($$y_{ctxt}$$) and one for each response candidate ($$y_{cand_i}$$), to then calculate the score of each with the dot product: $$y_{cand_i}\cdot y_{ctxt}$$. These embeddings representations are constructed as follows:

1.  Right side of Figure 1: It obtains the candidates embeddings  using a transformer encoder (BERT) and an aggregator function, that can be simply taking the classifier embedding $$C$$ of the output, or the average of the tokens. 
2.  Left side of Figure 1: It encodes the context using another transformer and then performing $$m$$ attentions (with $$m$$ a hyper parameter). Each attention (see the definition [here](https://cfierro94.github.io/nlp-summaries/attention-is-all-you-need#scaled-dot-product-attention)) uses as keys and values the transformer output and as query a learned code $$c_i$$ unique for each attention. It then computes another attention on top of those embeddings, where the query is the $$y_{cand_i}$$ and the keys and values are the output from the other attention $$y^i_{ctxt}$$. In equations:

$$
T(x) = (h_1, ..., h_N) \qquad \text{(Transformer output)}\\
$$

$$
y^i_{ctxt} = \sum_jw_j^{c_i}h_j \qquad  \text{, where:} \\
(w_1^{c_i}, ..., w_N^{c_i}) = \text{softmax}(c_i\cdot h_1, ..., c_i\cdot h_N) \\
$$

$$
y_{ctxt} = \sum_iw_i y^i_{ctxt} \qquad \text{, where:}\\
(w_1, ..., w_m) = \text{softmax}(y_{cand_i}\cdot y_{ctxt}^1, ..., y_{cand_i}\cdot y_{ctxt}^m)
$$

### Generator 

([Aswani et. al, 2017](https://arxiv.org/abs/1706.03762))

Standard Seq2seq model, like the transformer of "Attention is all you need" ([summary here](https://cfierro94.github.io/nlp-summaries/attention-is-all-you-need)) but way bigger (90M, 2.7B, 9.4B). In comparison, Meena Google's chatbot ([Adiwardana et. al, 2020](https://arxiv.org/abs/2001.09977)) has 2.7B parameters.

### Retrieve and refine

([Weston et al., 2018](https://arxiv.org/abs/1808.04776))

Trying to solve the problems of generator models (hallucinate knowledge, unable to read and access external knowledge, dull and repetitive responses). Here they mix the two models above appending to the input of a generator model the output of a retriever model (Figure 2), using a special separator token. They experiment with two types:

1. **Dialogue retrieval**: it uses the dialog history and it produces a response (Same retriever architecture)
2. **Knowledge retriever**: it retrieves from a large knowledge base, where the candidates are obtained from a TF-IDF-based inverted index lookup over a Wikipedia dump. For this case a transformer is additionally trained to decide when to add the knowledge retrieval and when not to (as some contexts do not require knowledge).

{% include image.html file="../assets/img/nlp-summary-03/retnrefine.png"
description="Figure 2. Retrieve and Refine architecture." zoom=55 %}

## Training objectives 

- <u>Retriever</u>: cross entropy over the $$y_{cand_i}$$ where $$y_{cand_1}$$ is the score of the correct response and the rest are negatives.
- <u>Generator</u>: standard maximum likelihood estimation (MLE)
- <u>Dialogue retrieval</u>: with MLE is not clear the relation between the retrieval response and the gold label (the correct answer). It has been proven that just using MLE makes the model ignore completely the retrieved utterance. Thus, here they replace the gold label with the retrieved utterance $$\alpha \%$$ of the times.
- <u>Knowledge retrieval</u>: here we can simply use MLE because the fine-tuning datasets used have a clear correspondence between the correct knowledge retrieval and response.

### Unlikelihood training

([Welleck et al., 2020](https://arxiv.org/pdf/1908.04319))

They also tried this objective because it was created to mitigate problems of MLE when training language models, such as repetition: using the same tokens more frequently than a human, and token distribution mismatch: using low frequency tokens (more specific tokens) too rarely compared to humans.

<u>Main idea:</u> to decrease the model’s probability of certain tokens, called negative candidates $$C_t$$. To do this, will add an expression to the MLE loss that we'll take these candidates into account, this is what we call unlikelihood loss:

$$
-\sum_{c\in C_t}\log(1-p_\theta(c|x_{< t}))
$$

Where $$p_\theta$$ is our language model predictions, and $$x_{< t}$$ is a sequence of $$t$$ tokens. As typically with losses, we have a negative logarithm that we will minimize, which is equivalent to maximizing the logarithm, therefore we'll be maximizing whatever is inside it. Since in this case we don't want the negative candidates ($$c$$) to be highly probable, we'll maximize the likelihood of not having them, so we'll maximize $$ 1-p_\theta(...) $$.

Thus, the actual training objective will be a mixture (gated by $$\alpha$$ hyper parameter) of the unlikelihood of bad candidates and the likelihood of the next token:

$$
-\alpha \underbrace{-\sum_{c\in C_t}\log(1-p_\theta(c|x_{<t}))}_{\text{unlikelihood}} \underbrace{- \log(p_\theta(x_t|x_{<t}))}_{\text{likelihood}}
$$

 In the paper they defined the set of bad candidates as the tokens that were generated more frequently by the model than by humans. To measure these frequencies they kept a running count of the distribution of the tokens generated by the model and they compared it to the distribution of the gold responses.

## Decoding

They tried different decoding strategies:

1. **Beam search** ([summary here](https://cfierro94.github.io/nlp-deep-dive/attention-is-all-you-need#beam-search))
2. **Top-k sampling**: at each time step the word $$i$$ is selected by sampling from the k (=10) most likely candidates from the model distribution.
3. **Sample-and-rank sampling**: $$N$$ independent sentences are sampled (following the model probabilities) and then the one with the highest probability is selected.

They also tried additional constraints for the decoding process:

1. **Minimum length**: Force the model to produce an answer of a defined length.
2. **Predictive length**: Predict (with a retriever model) the minimum length of the answer (e.g., <10, <20, <30, >30 tokens). And then we do the same as in 1.
3. **Beam blocking**: Force the model to not produce in the next utterance a trigram (group of 3 words) that's in the input or in the utterance itself. That can be achieved by setting to 0 the probability of the words that would create a trigram that already exists.

## Training data

Train:

- Pushshift.io Reddit: reddit discussions covering a **vast range of topics**.

Two-way conversational data to fine tune the models:

- ConvAI2 dataset ([Zhang et al., 2018](https://arxiv.org/abs/1801.07243)) focuses on **personality** and engaging the other speaker. It gives a persona description to the speaker (which is concatenated to the history to use it as input in the model).
-  Empathetic Dialogues ([Rashkin et al., 2018](https://arxiv.org/abs/1811.00207)) focuses on **empathy**.
- Wizard of Wikipedia ([Dinan et al., 2018](https://arxiv.org/abs/1811.01241)) focuses on **knowledge**. 
- Blended Skill Talk ([Smith et al., 2020](https://arxiv.org/pdf/2004.08449)) provides a dataset that focuses on blending all the previous skills. This is constructed with one human speaking freely (using its persona) and the other one guided, that is he/she has to choose an utterance response from 3 different possibilities constructed by a model trained in each of the three previous datasets.

## Evaluation methods

#### ACUTE-eval

This is a manual evaluation where two different dialogues (between a model an a human) are presented to a person that needs to answer:

- “Who would you prefer to talk to for a long conversation?” (Engagingness)
-  “Which speaker sounds more human?” (Humanness)

#### Self-Chat ACUTE-Eval

Same as ACUTE-eval but we the dialogues are generated by the model talking to itself instead of a human.

## Results

The results are comparisons in the votes given for each question presented above. In the paper they say that some results are "not significant", which basically means that given the amount of answers collected and the votes in each side, is not certain if one is better than the other, as in the difference could be noise of the measure.

### Results of Self-Chat ACUTE-Eval

- When comparing the **3 models** using standard beam search (beam size 10, no minimum beam decoding constraint, but with context and response 3-gram blocking), the results are Retriever > RetNRef > Generator.
- When comparing <u>decoding choices</u>: 
  - In terms of **minimum length**: The best results were encountered when setting a minimum length of 20 or 40, or when predicting the minimum length using the buckets 10,20,30,40.
  - In terms of **beam blocking**: Blocking both context and response 3-grams during generation gives highest scores, however, the differences were not significant.
  - Comparing different **beam sizes and sampling methods**, it appears that a beam value of 10 is superior to 1 or 30, and a 10 size beam is on par with sampling methods.
- Larger models perform better
- Fine tuning in the 4 extra datasets give huge improvements
- Using the persona context (description about a specific persona) after having fine tuning gives a little improvement compared to not using them.
- Unlikelihood training has a small gain (although it's not statistically significant). Notice that the conversations in these experiments are short so maybe the advantages of this training objective are not totally exploited.

### Results of ACUTE-eval

Results of conversations of 14 turns between humans-chatbot.

- Comparing the 3 models with the improved decoding strategy (beam size 10, minimum length 20, blocking context and response) the results were: RetNRef > Generator > Retriever.
- Comparing to Meena ([Adiwardana et. al, 2020](https://arxiv.org/abs/2001.09977)):
  - In the engagingness question the generative model of the same size is better 75% of the times.
  - In the humanness question the generative model of the same size is better 65% of the times, and the generative of the same size trained with unlikelihood is better 70% of the times.

When comparing one human-chatbot dialogue to a human-human dialogue: the results that are statistically significant show that the models in this paper are 37% of the times better than human-human dialogues in the engagingness question. Additionally, the generative model is 49% of the times better in the same question, but this is not statistically significant. Even though this result sounds promising, the model is not this close to a human dialogue, below we can see the flaws that the authors presented and that are not really measured by this evaluation.

### Failure cases

- **Words repetition.** The minimum length helps to create more detailed messages, but the core problem still remains. Some 3-grams were over-expressed compared to human-human conversations, such as: “do you like”, “lot of fun”, “have any hobbies”.
  - Evaluation problem: the current evaluation does not seem to expose this as boring because the conversations are short and are evaluated separately.
- **Ideas repetition.** Beam blocking helps with this issue, but it can be seen that the model has a tendency to repeat what the other part says, if the human says he/she has a dog then the bot repeats that it has one too, the chatbot likes the same bands as you, etc.
- **Forgetfulness.** The model does not link correctly to past statements, for example you tell the model you have a dog, but then later in the conversation it asks what pets do you have.
- **Contradiction.** It makes contradictions linked to overall knowledge, for example it says it lives in the MidWest and then it specifies it lives in Georgia (which is not in the midwest).
- **Knowledge**. They observed that the models often switch topics, avoiding the challenge of going “deeper". The reading of knowledge only hurt the model in the evaluation setup, possibly due to:
  - The model attempts to use knowledge when there is no need, or using it incorrectly.
  - Evaluation problem: deeper knowledge is not really required in this setup, since the dialogues are short and tend to cover only shallow topics whereby the speakers get to know each other.
- **Context length.** The models in this paper have a hard limit of 128 tokens, there's been some research in this problem but it would need another setup to be evaluated (with dialogues longer than 14 turns).
- **Deeper understanding.** These models cannot be taught a concept through further conversation, so as-is they will always be stunted. See fun examples in Figure 3.

{% include image.html file="../assets/img/nlp-summary-03/dialogues.png"
description="Figure 3. Pun dialogues failures." zoom=100 %}

## What's next!

This paper shows a really robust and advanced chatbot, however it also presented a lot of remaining challenges to really be near a human-like bot. If you're interested in one of the challenges presented in this summary go and read the paper! 😀 Because it cites work that is trying to overcome these issues.