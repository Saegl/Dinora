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

There is a tool for tree visualization, look more [here](https://dinora.readthedocs.io/en/latest/treeviz.html)

![Treeviz Visualization](assets/treeviz-example/state.svg)

# Acknowledgements

This engine based on https://github.com/Zeta36/chess-alpha-zero and 
https://github.com/dkappe/a0lite and Alphazero from Deepmind.

A lot of tutorials about chess engines from https://www.chessprogramming.org/ was super helpful.
