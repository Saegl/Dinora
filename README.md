# Dinora

[![Documentation Status](https://readthedocs.org/projects/dinora/badge/?version=latest)](https://dinora.readthedocs.io/en/latest/?badge=latest)


[Documentation](https://dinora.readthedocs.io/en/latest/) | [Installation](https://dinora.readthedocs.io/en/latest/installation.html)

Dinora is alphazero-like chess engine. It uses 
keras/tensorflow for position evaluation and Monte Carlo Tree Search for 
calculating best move.

### Features
- Working chess engine
- Minimal example of alpazero-like engine NN + MCTS
- All code included in this repo - for playing and training
- Everything written in python

## Status
You can play against Dinora in standard chess variation, with or without increment.
I assume engine strength is about 1400 Lichess Elo, I evaluate engine rating 
basing on a few games against me, so it's not accurate.  
You can see example game below  
(10+0) Dinora (100-200 nodes in search) vs Me (2200 Rapid Lichess rating)  

<img src="https://github.com/Saegl/dinora/raw/main/assets/gif/gfychess-example.gif" width="350">

## Tree Visualization

There is a tool for tree visualization.

![Treeviz Visualization](assets/treeviz-example/state.png)

Original vector images can be found at (/assets/treeviz-example/)  
To generate new visualizations see
```python -m dinora treeviz --help```

# Acknowledgements

- [AlphaZero](https://deepmind.google/discover/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) Original AlphaZero resources
- [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)
First/(one of the first) open source alphazero implementation in python
- [dkappe/a0lite](https://github.com/dkappe/a0lite) NN + MCTS in 95 lines of
python code
- [Chess Wiki](https://www.chessprogramming.org) Good resource on chess engines
  in general
- [int8 MCTS article](https://int8.io/monte-carlo-tree-search-beginners-guide/)
  Intro to MCTS blog post
- [Deep Dive MCTS](https://www.moderndescartes.com/essays/deep_dive_mcts/)
Another great article on MCTS
- [Stockfish](https://stockfishchess.org/) Strongest chess engine, used here for
  test / training data annotation
- [Leela Chess Zero](https://lczero.org/) If you really want to use AlphaZero
inspired chess engine this is the real one
- [Pytorch](https://pytorch.org/) Library to train neural networks
- [Python chess](https://python-chess.readthedocs.io/en/latest/) Library for
chess (rules, legal moves generator, pgn reader/writer, UCI interface)

