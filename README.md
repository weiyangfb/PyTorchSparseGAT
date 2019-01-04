# Pytorch Graph Attention Network

This is a PyTorch implementation of the Graph Attention Network (GAT)
model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

The repo has been forked initially from https://github.com/Diego999/pyGAT by
simplifying class `SpGraphAttentionLayer` using PyTorch sparse operators.

Please cite the following:
```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```

The branch **master** contains the implementation using sparse operators.
