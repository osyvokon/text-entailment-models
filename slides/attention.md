
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

Read source sentence one token a time. Often, token is a word,
but also can be a character or sub-word.

Compute the source sentence embedding from the sequence of hidden states
$$h_1 ... h_n$$.
In the simpliest case (which works reasonably well), just take

---

# Encoder-decoder

![:scale 100%](img/encoder-decoder.png)

State bottleneck: 

# Content-based neural attention


.left-column30[![:scale 80%](img/bahdanau.jpeg)]

.right-column70
#### Neural Machine Translation by Jointly Learning to Align and Translate

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.

2015

]

???

* ICLR 2015 paper
* Belarusian State University

---

# Attention intuition

* Simple random access memory
* Can access source states as needed


---

# Other Applications

* Caption generationA [Show, Attend and Tell]


# Gated Attention

## Gated-Attention Readers for Text Comprehension

[Dhingra, 2017]

???

* ICLR 2017 paper.
* State-of-the-art on cloze-style questions.


---

# Attention-over-Attention

Introduced in [Cui, Chen, 2016] for cloze-style
reading comprehension task.


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
