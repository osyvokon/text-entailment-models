
class: center, middle

# Neural Attention Mechanism in NLP

Oleksiy Syvokon

research engineer @ Grammarly

---

# Why bother?

.right[
>>> I am very excited by the recently introduced
>>> *attention models*, due to their simplicity and due to
>>> the fact that they work so well.
>>>
>>> — <cite>Ilya Sutskever</cite>
]

???

Ilya is a Co-Founder and Research Director at OpenAI

--
1. Achieve state-of-the-art results on many NLP applications:

    * Neural translation
    * Text summarization
    * Text entailment
    * Text comprehension (question answering)
--

2. Easy to implement and grasp
--

3. Useful visualisations.


---

# Agenda

1. Prerequisites

    - sentence embeddings
    - sequence-to-sequence 101
    - encoder-decoder bottleneck

--
2. Attention mechanism

    - neural translation task
    - text entailment task
--

3. Attention advances:

    - soft vs hard
    - local vs global
    - hierarchical
    - attention-over-attention
    - gated

---

# Sequence-to-sequence models


---

# Neural Machine Translation

Neural Machine Translation (NMT) aims to model the
entire MT process via **one** big artificial neural
network.

---

# Encoder-decoder

![:scale 100%](img/encoder-decoder.png)


???

Read the source sentence one token a time. Often, token is a word,
but also can be a character or sub-word.

Compute the source sentence embedding from the sequence of hidden states
$h_1 ... h_n$.  In the simpliest case (which works reasonably well), just take
the last state,  $h_5$ in this example. Other options include convolving
over hidden states, or combining them in a recursive-tree manner.

What matters, is that in the end we have a single vector `h_6` that represents
(encodes) the whole sentence. This sentence embedding can be somewhat
similar to the familiar word embeddings. For example, sentences that have
similar meaning will likely have representations that are close to each other.

Having sentence embedding in hand, we can use it for banch of tasks.
We can feed it into a classifier to get sentence category, like sentiment.
For a pair of sentences, we can concatenate the two states and again feed that
to a classifer, for example, to predict entailment relations.

For encoder-decoder architectures, we use that embedding to generate an
output sequence. We have another recurrent network (say, LSTM), that starts
with its state initialized by $h_6$. From that, it generates a new hidden
state, $h_7$, and predicts the first token by generating a softmaxed
distribution over the vocubalary. Let's say, we've picked the $argmax(y_7)$,
everything went well and the first word was "кіт" in our case, just as expected.

On the next step, we feed the predicted $y$ and the previous hidden state
$h_7$ to generate the probabilty distribution for the second word.
We continue until we see the end-of-sentence symbol.

That is a general intuition of how neural translation works. In practice,
of course, it is slighetly more complicated. The architecture has several
layers. Normally, bi-directional LSTM are used, so that we encode source sentence
*twice*, the original and the reversed version, and the concatenate the two
state vectors. On the decoder stage, it is common to do a beam search
(with a relatevily small beam size of 3 to 15) instead of a greedy search
that I've described. However, even the simple architecture will be able
to give more-or-less meaningful results.


---

# Encoder-decoder: state bottleneck

![:scale 100%](img/encoder-decoder-state-bottleneck.png)

State bottleneck: 

???

The idea of representing the whole sentence meaning in a single fixed-sized
vector is very appealing, but it also means that the decoder must start
having only a fixed-sized vector at hand. Unfortunately, it doesn't work that
well with long sentences. Intuitevely, when we try to cramp the long and/or
complicated sentence into a fixed-sized vector, we will inevitably loose
details that are important for tasks like translation.

That is where attention mechanism comes into play.

What we humans do when translating, is regularaly look onto the source sentence
a couple words a time. We don't just read the source sentence once and then
throw it away. We keep it around and concentrate on relevant parts of it when
needed. That is the basic idea behind the attention mechanism: let's keep the
source hidden states pool and draw relevant parts of it on the decoder stage.

---

# Attention intuition

![:scale 100%](img/encoder-decoder-highilght-cat.png)

???

For example, when at the start of the sentence, I will look at the word "cat".

---

# Attention intuition

![:scale 100%](img/encoder-decoder-highilght-on.png)

???

Then I move to the next word, and only interested in the second word (and maybe
also in some surroinding context as well as in the general context of what this
sentence is all about).

---

# Attention intuition

![:scale 100%](img/encoder-decoder-highilght-the-mat.png)

???

Finally, I get to the last part.

---


# Attention intuition: words alignment

.right[.medium[*NMT by Jointly Learning to Align and Translate ([Bahdanau, 2014](https://arxiv.org/abs/1409.0473))*]]

![:scale 50%](img/bahdanau-alignment.png)


???

Turns out, machine translation has known this long before recurrent nets
revivals. They call it "words alignment", and it can be represented by
the alignment matrix.

---



# Content-based neural attention


.left-column30[![:scale 90%](img/bahdanau.jpeg)]

.right-column70[
#### Neural Machine Translation by Jointly Learning to Align and Translate

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.

2015

]

???

* ICLR 2015 paper
* Graduated Belarusian State University
* 1215 citations so far (and counting) -- every paper about attention cites this.
* Started the attention boom
* Simple and effective techinique

---

## Encoder-decoder, no attention

.small[.left-column30[
1. Run RNN over the input:
<div>$$h_t = \text{RNN}(\text{embed}(x_t, h_{t-1}))$$</div>

2.  Concatenate hidden state vectors into a matrix $H$:
  $$ H = \left[ \begin{array}{c} h_1 & h_2 & h_3 & ... & h_n \end{array} \right] $$

3. Encode input sequence into a vector $c$:
  $$c = q(H)$$

  For example,
  $$c = h_n$$

4. Predict the next output word:
<div>$$y_t = g(y_{t-1}, s_{t-1}, c)$$</div>

]]

.right-column70[.right[![:scale 100%](img/encoder-decoder.png)]]

???

Okay, let's have a more formalized look at what we've just discussed.
We have an input sequence of words, $x_1 ... x_n$. We map this input into
embedding vectors. Then we we run recurrent neural, producing hidden states
$h_1 ... h_n$ along the way.

For the *non-attention* case, we then summary the whole source sentence into
a single vector $\mathbf{c}\$, commonly by just taking the last hidden state
and discarding all others.

We use that context vector to predict output sequence one word a time.

---

## Encoder-decoder, content-based attention

.small[.left-column30[
1. Run RNN over the input:
<div>$$h_t = \text{RNN}(\text{embed}(x_t, h_{t-1}))$$</div>

2. Concatenate hidden state vectors into a matrix $H$:
  $$ H = \left[ \begin{array}{c} h_1 & h_2 & h_3 & ... & h_n \end{array} \right] $$

3. Generate new context vector for each output word:
<div>
  $$c_i = H \mathbf{\alpha}_t$$
</div>

4. Predict the next output word:
<div>$$y_t = g(y_{t-1}, s_{t-1}, \color{red}{c_i})$$</div>

]]

.right-column70[.right[![:scale 100%](img/encoder-decoder.png)]]

???

Now, let's see what changes with the attention.

As before, we start by encoding the source sentence with RNN (often, with a
bi-directional RNN, but that doesn't really matter). This time we keep around
hidden state vectors corresponding to the input words. Let's concatenate them
in a matrix $H$.

Now comes the difference. Instead of computing sentence embedding (which is
also called a *context vector*) just once, we will do it again and again
for each of the output words. Decoder will be getting a context that is
specifically crafted for predicting each output.

What does this context vector look like? Well, it's just a weighted combination
of the input states ${h_1, h_2, ..., h_n}$:

<div>
  $$\mathbf{c}_i = H \mathbf{\alpha}_t$$
</div>


---

## Attention and context vectors

.left-column50[

Context vector $\mathbf{c}_i$ is a weighted sum over the source sentence:

<div>
$$\begin{aligned}

\mathbf{c}_i &= H \mathbf{\alpha}_t 
 \\
 &= \sum_{j=1}^n h_j \alpha_{ij}

\end{aligned}$$
</div>

Attention vector $\mathbf{\alpha_t}$ tells us how much should we focus on a
particular source word at a time step $t$:
]

.right-column50[![:scale 100%](img/attention-vector-1.png)]


---

## Attention and context vectors

.left-column50[

Context vector $\mathbf{c}_i$ is a weighted sum over the source sentence:

<div>
$$\begin{aligned}

\mathbf{c}_i &= H \mathbf{\alpha}_t 
 \\
 &= \sum_{j=1}^n h_j \alpha_{ij}

\end{aligned}$$
</div>

Attention vector $\mathbf{\alpha_t}$ tells us how much should we focus on a
particular source word at a time step $t$:
]

.right-column50[![:scale 100%](img/attention-vector-2.png)]

---

## Attention and context vectors

.left-column50[

Context vector $\mathbf{c}_i$ is a weighted sum over the source sentence:

<div>
$$\begin{aligned}

\mathbf{c}_i &= H \mathbf{\alpha}_t 
 \\
 &= \sum_{j=1}^n h_j \alpha_{ij}

\end{aligned}$$
</div>

Attention vector $\mathbf{\alpha_t}$ tells us how much should we focus on a
particular source word at a time step $t$:
]

.right-column50[![:scale 100%](img/attention-vector-3.png)]

---

class: left

## How to compute attention


1. Compute the decoder's hidden state:

   <div>$$h_t = \text{RNN}([\text{embed}(y_{t-1}); c_{t-1}], h_{t-1})$$</div>

--
2. Calculate an attention score $a_t$:

   <div>$$a_{t,j} = \text{attention\_score}(h_j^{(src)}, h_t^{(dest)}) $$</div>

--
3. Normalize that:

    <div>$$\mathbf{\alpha}_t = \text{softmax}(a_t)$$</div>

???

From where do we get this $\alpha_t$? 

Firstly, compute the decoder's hidden state $h\_t$ . Note, that we expand its
input to include the context vector $c_{t-1}$ from the previous step.

`attention_score` can be any function that takes two vectors as input
and outputs a score about how much we should focus on this particular
word encoding $h_j^{src}$. We will discuss this in more details in a second.

Finally, we normalize attention scores to get the properties we need:
it should sum to one and all its elements should be between 0 and 1.
This is what we need to get linear combination of the input hidden vectors.

---

## Attention step-by-step

* Encode the source sentence.

![:scale 100%](img/encoder-decoder-attention-overview 0.png)

---

## Attention step-by-step

* Compute the first hidden state of the decoder.

![:scale 100%](img/encoder-decoder-attention-overview 1.png)

???

  - start-of-sentence token as a word input
  - null context vector
  - previous state taken to be the source sentence encoding

---

## Attention step-by-step

* Compute the attention vector.

![:scale 100%](img/encoder-decoder-attention-overview 2.png)

---

## Attention step-by-step

* Compute the context vector.

![:scale 100%](img/encoder-decoder-attention-overview 3.png)

---

## Attention step-by-step

* Predict the first word

![:scale 100%](img/encoder-decoder-attention-overview 4.png)

---

## Attention step-by-step

* Use that context vector and the predicted word to get next hidden state

![:scale 100%](img/encoder-decoder-attention-overview 5.png)

---

## Attention step-by-step

* Repeat

![:scale 100%](img/encoder-decoder-attention-overview 6.png)

---

## Recap

1. What problem attention solves?

2. What is a context vector $c_i$?

3. What is attention vector $\alpha_t$?

4. How to compute attention vector?  (that we don't know yet)

???

By now, we should have a solid intuition of what attention mechanism tries
to achieve. We also saw how it's used to calculate context vectors and how
we pass those into the decoder RNN. This remaining missing piece is how
to get the attention score.

---

## Ways to compute attention scores

### Dot product:

<div>$$ \text{attention\_score}(h_j^{(src)}, h_t^{(dest)}) = h_j^{(src)\intercal}  h_t^{(dest)} $$</div>

.medium[
Pros:
  * adds no additional parameters
  * simple and fast

Cons:
  * forces input and output hidden vectors to be in the same space
]

???
The simplest option is to measure similarity between query and keys by taking
dot product (скалярний добуток) between two vectors. The advantage of this
model is that it adds no additional parameters. Disadvantage is that it's not
flexible, forcing input and output hidden states be in the same space.  In
order for dot product to be high, vectors must lay close to each other.

---

## Ways to compute attention scores

### Bilinear functions:

<div>$$ \text{attention\_score}(h_j^{(src)}, h_t^{(dest)}) = h_j^{(src)\intercal} W_a  h_t^{(dest)} $$</div>

.medium[

Pros:
  * more flexible

Cons:
  * adds quite a few parameters
]

???
By performing a linear transform parametrized by $W_a$ we relax the
restriction that the source and target embeddings msut be in the same
space. On the other hand, it adds quite a few parameters.

---

## Ways to compute attention scores

### Multi-layer perceptrons:

<div>$$ \text{attention\_score}(h_j^{(src)}, h_t^{(dest)}) = w_{a2}^\intercal \tanh(W_{a1} [h_j^{(src)}; h_t^{(dest)}])$$</div>

.medium[

Pros:
  * flexible
  * fewer parameters

Cons:
  * doesn't fit complex structures
]


???
Multi-layer perceptron was the method originally employed by Bahdanau.
This is more flexible that the dot product method, usually has fewer
parameters of the three, and generally provides good results.

---

## Attention scores

### Advanced methods

* Recurrent NNs

* Tree-structured networks

* Convolutional NNs

* Structured models

...and more


???

RNN attention: very complicated, +3 BLUU for English-German

>>> Knowing which words have been attended to in previous time steps while generating a translation is a rich source of
information for predicting what words will be attended to in the future. We
improve upon the attention model of Bahdanau et al. (2014) by explicitly
modeling the relationship between previous and subsequent attention levels for
each word using one recurrent network per input word. This architecture easily
captures informative features, such as fertility and regularities in relative
distortion. In experiments, we show our parameterization of attention improves
translation quality.

---

## Image Caption Generation

.right[.medium[*Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* ([Xu et al., 2016](https://arxiv.org/abs/1502.03044))]]

![:scale 80%](img/show-attend-tell-arch.png)

---

## Soft vs. Hard attention

.right[.medium[*Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* ([Xu et al., 2016](https://arxiv.org/abs/1502.03044))]]

Soft: weighted sum of features.

Differentiable, but wasteful.

![:scale 80%](img/show-attend-tell-soft.png)

???

"Show, Attend and Tell" introduces two modes of attention.

Soft is just a weighted sum of features. It’s easy to compute because it’s
differentiable. At the same time, on each step we attend the whole input
sequence, so that's computationally expensive and comes against intuition
(humans don't do it that way).

---

## Soft vs. Hard attention

.right[.medium[*Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* ([Xu et al., 2016](https://arxiv.org/abs/1502.03044))]]

Hard: take only one feature according to $\alpha$.

Nnon-differentiable, but faster on prediction time.

![:scale 80%](img/show-attend-tell-hard.png)

???

Hard attention means take only one feature according to alpha. Derivative is
zero almost everywhere, so cannot backpropagate and need to train with
methods like reinforcement learning -- much slower to train than with
backprop.

---


# Gated Attention

## Gated-Attention Readers for Text Comprehension

[Dhingra, 2017]

???

* ICLR 2017 paper.
* State-of-the-art on cloze-style questions.


---

## Neural Machine Translation with Recurrent Attention Modeling


---

# Attention-over-Attention

Introduced in [Cui, Chen, 2016] for cloze-style reading comprehension task.

---

# Hierarchical Attention Networks

.right[.medium[*Hierarchical Attention Networks for Document Classification ([Yang, 2016](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)) *]]


![:scale 50%](./img/hierarchical-att.png)

???
Hierarchical Attention Networks for Document Classification

---

# Hierarchical Attention Networks

.right[.medium[*Hierarchical Attention Networks for Document Classification ([Yang, 2016](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)) *]]

SOTA on document classification.


| Method            | Yelp'15 | IMDB |
| ----------------- | ------- | ---- |
| SVM + Features    | 62.4    | 40.5 |
| LSTM              | 58.2    | -    |
| LSTM-GRNN         | 67.6    | 45.3 |
| Hierarchical Ave  | 69.9    | 47.8 |
| Hierarchical ATT  | **71.0**  | **49.4** |

---

# Hierarchical Attention Networks

.right[.medium[*Hierarchical Attention Networks for Document Classification ([Yang, 2016](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)) *]]

Predicting YELP rating:

![:scale 60%](./img/hierarchical-yelp.png)

---

# Attention networks results


---

# Stanford Natural Language Inference Corpus

**ENTAILMENT**:
  * A man rides a bike on a snow covered road.
  * A man is outside.


**NEUTRAL**:
  * 2 female babies eating chips.
  * Two female babies are enjoying chips.


**CONTRADICTION**
  * A man in an apron shopping at a market.
  * A man in an apron is preparing dinner.
