# Training

For training you need [wandb account](https://wandb.ai), to store various
artifacts (datasets, weights, graphs). Fully local training is not supported

Also, you need access to Nvidia GPU

Training is 3 steps:
1. Dataset preprocessing
2. Neural network training on GPU
3. Chess engines minimatch to evaluate strength


## Dataset

To start training you need games dataset.  
You can make your own dataset (takes hours of preprocessing, depending on
dataset size and stockfish annotations depth),
or [take one of my datasets](https://wandb.ai/saegl/dinora-chess/artifacts/dataset/traindocs/v0).
To take my dataset you need a wandb label for now,
like "saegl/dinora-chess/traindocs:v0", that will be placed in training config later.


### Create new dataset

(Skip this step if you took my label)  

For this step you need spare computer for a lot of time, no GPU needed at this
point, cheap VPS or old computer is enough.
To create dataset you need [PGN files](https://en.wikipedia.org/wiki/Portable_Game_Notation).

You can take them from these places
- [Lichess Database](https://database.lichess.org/): Games played on lichess.org between
  humans
- [Lichess Elite](https://database.nikonoel.fr/): Games from lichess players but
  filtered to contain only (2400 vs 2200 Elo)
- [Leela Standard Dataset](https://lczero.org/blog/2018/09/a-standard-dataset/):
  Chess analog to MNIST, 2.5M games between chess engines
- Or any other PGN file with at least 100k games

Before training PGN need to be converted to numpy arrays.  

Prepare folder of PGNs like this:
```
example_dataset/1.pgn
example_dataset/2.pgn
example_dataset/3.pgn
...
```

You can use any name instead of `example_dataset`. Each pgn should have about 10k
games. Existing PGN tools will help, for example:

Let's take games from Lichess Elite


```bash
wget https://database.nikonoel.fr/lichess_elite_2022-04.zip
unzip lichess_elite_2022-04.zip
# Now we have single lichess_elite_2022-04.zip with 380k games

# Download pgn-extract to split this file
# Newer versions https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/
wget https://www.cs.kent.ac.uk/~djb/pgn-extract/pgn-extract-25-01.zip
unzip pgn-extract-25-01.zip
cd pgn-extract-25-01.zip
make
cd ..

mkdir example_dataset
cd example_dataset
# Split to 1.pgn 2.pgn ... each 10k games
../pgn-extract/pgn-extract ../lichess_elite_2022-04.pgn -#10000

# I actually included only {1..16}.pgn for this example
# rm {17..34}.pgn
cd ..
```

Once you have this `example_dataset` folder of PGNs, convert it to numpy arrays
and upload to wandb

```bash
mkdir dataset_np
# Took 40 minutes on my computer, 186MB of disk space
uv run python -m dinora make_dataset example_dataset/ dataset_np/ --q-nodes 100

# login to wandb first
wandb login
# Upload dataset to wandb
uv run python -m dinora upload_dataset dataset_np/ "traindocs"

# In my case, uploaded to "saegl/dinora-chess/traindocs:v0" wandb label
```

## Train

For this step you need Nvidia GPU. You can train from terminal or jupyter
notebook. In both cases progress will be tracked by wandb.

### Prepare config

Example configs are in `configs/train/` folder

Annotated example below (don't include comments in actual config)  
Or read `src/dinora/train/fit.py`
```jsonc
{
    "matmul_precision": "medium",
    "max_time": null,
    "max_epochs": -1,
    // wandb dataset label mentioned before
    "dataset_label": "saegl/dinora-chess/traindocs:v0",
    // position evaluation is combination of z and q
    "z_weight": 0.0, // Train from game outcome
    "q_weight": 1.0, // Train from Stockfish annotations

    "tune_batch": true, // Search for max batch before training
    "batch_size": 512, // overwritten in this case

    "tune_learning_rate": true, // Search for best lr
    "learning_rate": 0.6918309709189363, // overwritten in this case
    "lr_scheduler_gamma": 1.0, // lr decay, here is no effect (multiple by 1)
    "lr_scheduler_freq": 30000,

    // pytorch lightning checkpointing, was broken last time I check
    // use `enable_validation_checkpointer` below
    "enable_checkpointing": true,
    "checkpoint_train_time_interval": {"minutes": 30},

    // Those will render nice stats in wandb
    "enable_sample_game_generator": true,
    "enable_boards_evaluator": true,
    "enable_validation_checkpointer": true,

    "log_every_n_steps": 100,

    // Run validation 2 times in epoch, since epochs are usually huge
    "val_check_interval": 0.5,

    "limit_train_batches": null,
    "limit_val_batches": null,
    "limit_test_batches": null,

    // hyperparams of model
    "model_type": "alphanet",

    "res_channels": 256,
    "res_blocks": 19,
    "policy_channels": 64,
    "value_channels": 8,
    "value_lin_channels": 256
}
```

### From terminal

```bash
uv run python -m dinora.train.fit <path-to-config>
```

### From jupyter notebook

This is useful when you want to run in services like Google Colab / Kaggle /
Vast.ai. They have jupyter notebooks with `torch` preinstalled,
everything else will be installed from notebook itself.

Just copy `jupyter/train.ipynb` to service, update configs in notebook and run


### Charts

Both method will print wandb url with output graphs and weights,
[like this one](https://wandb.ai/saegl/dinora-chess/runs/ozs75vi1/workspace?nw=nwusersaegl)

## Evaluation

Besides train/validation loss graphs, to measure actual strength of resulted
neural networks you need to run little minimatch against stockfish. Look at
`uv python -m elofish --help`

TODO: Add more info about elofish usage and configuration

