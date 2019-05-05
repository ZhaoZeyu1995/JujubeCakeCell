<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# JujubeCakeCell

## Introduction

This is a nested LSTM RNN cell, named JujubeCakeCell. It has a structure as shown in this figure

![JujubeCakeCell](https://ws4.sinaimg.cn/large/006tNc79ly1g2qhhqnznyj30mc0dntaa.jpg)

In the process of JujubeCakeCell, it has a chain of LSTM cell and it collect the output of some LSTM cell to compose a JujubeCakeCell state vector. As for other aspects, JujubeCakeCell has a similar structure as LSTM cell.

In particular, an LSTM cell can be described by using these functions.

$
a+b
$
