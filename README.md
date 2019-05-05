# JujubeCakeCell

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## Introduction

This is a nested LSTM RNN cell, named JujubeCakeCell. It has a structure as shown in this figure

![JujubeCakeCell](https://ws4.sinaimg.cn/large/006tNc79ly1g2qhhqnznyj30mc0dntaa.jpg)

In the process of JujubeCakeCell, it has a chain of LSTM cell and it collect the output of some LSTM cell to compose a JujubeCakeCell state vector. As for other aspects, JujubeCakeCell has a similar structure as LSTM cell.

In particular, an LSTM cell can be described by using these functions.

![LSTMEquation1]\(&lt;<http://chart.googleapis.com/chart?cht=tx&chl=\mathbf{f}_t> = \\sigma(\\mathbf{x}_t\\mathbf{W}_{xf}+\\mathbf{h}_{t-1}\\mathbf{W}_{hf}+\\mathbf{b}\_{f}))

![LSTMEquation2]\(&lt;<http://chart.googleapis.com/chart?cht=tx&chl=\mathbf{o}_t> = \\sigma(\\mathbf{x}_t\\mathbf{W}_{xo}+\\mathbf{h}_{t-1}\\mathbf{W}_{ho}+\\mathbf{b}\_o))

![LSTMEquation3]\(&lt;<http://chart.googleapis.com/chart?cht=tx&chl=\mathbf{i}_t> = \\sigma(\\mathbf{x}_t\\mathbf{W}_{xi}+\\mathbf{h}_{t-1}\\mathbf{W}_{hi}+\\mathbf{b}\_i))

$$
\\mathbf{f}_t = \\sigma(\\mathbf{x}\_t\\mathbf{W}_{xf}+\\mathbf{h}_{t-1}\\mathbf{W}_{hf}+\\mathbf{b}\_{f})
$$
