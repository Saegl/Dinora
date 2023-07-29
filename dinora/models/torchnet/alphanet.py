"""
Alphazero network implementation getted from paper:
Mastering the game of Go without human knowledge, Silver et al.

'The input features st are processed by a residual tower that consists of a single
convolutional block followed by either 19 or 39 residual blocks.
The convolutional block applies the following modules:
(1) A convolution of 256 filters of kernel size 3 ×​3 with stride 1
(2) Batch normalization 18
(3) A rectifier nonlinearity
Each residual block applies the following modules sequentially to its input:
(1) A convolution of 256 filters of kernel size 3 ×​3 with stride 1
(2) Batch normalization
(3) A rectifier nonlinearity
(4) A convolution of 256 filters of kernel size 3 ×​3 with stride 1
(5) Batch normalization
(6) A skip connection that adds the input to the block
(7) A rectifier nonlinearity
The output of the residual tower is passed into two separate ‘heads’ for
computing the policy and value. The policy head applies the following modules:
(1) A convolution of 2 filters of kernel size 1 ×​1 with stride 1
(2) Batch normalization
(3) A rectifier nonlinearity
(4) A fully connected linear layer that outputs a vector of size 192 +​ 1 =​ 362,
corresponding to logistic probabilities for all intersections and the pass move
The value head applies the following modules:
(1) A convolution of 1 filter of kernel size 1 ×​1 with stride 1
(2) Batch normalization
(3) A rectifier nonlinearity
(4) A fully connected linear layer to a hidden layer of size 256
(5) A rectifier nonlinearity
(6) A fully connected linear layer to a scalar
(7) A tanh nonlinearity outputting a scalar in the range [−​1, 1]
The overall network depth, in the 20- or 40-block network, is 39 or 79 parameterized layers, respectively,
for the residual tower, plus an additional 2 layers for the policy head and 3 layers for the value head.
We note that a different variant of residual networks was simultaneously applied
to computer Go33 and achieved an amateur dan-level performance; however, this
was restricted to a single-headed policy network trained solely by supervised learning.

Neural network architecture comparison. Figure 4 shows the results of a comparison between network architectures.
Specifically, we compared four different neural networks:
(1) dual–res: the network contains a 20-block residual tower, as described above,
followed by both a policy head and a value head. This is the architecture used in AlphaGo Zero.
(2) sep–res: the network contains two 20-block residual towers. The first tower
is followed by a policy head and the second tower is followed by a value head.
(3) dual–conv: the network contains a non-residual tower of 12 convolutional
blocks, followed by both a policy head and a value head.
(4) sep–conv: the network contains two non-residual towers of 12 convolutional
blocks. The first tower is followed by a policy head and the second tower is followed
by a value head. This is the architecture used in AlphaGo Lee.
Each network was trained on a fixed dataset containing the final 2 million
games of self-play data generated by a previous run of AlphaGo Zero, using
stochastic gradient descent with the annealing rate, momentum and regularization hyperparameters described for
the supervised learning experiment; however, cross-entropy and MSE components were weighted equally,
since more data was available.'
"""
import chess
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import lightning.pytorch as pl

from dinora.board_representation2 import board_to_tensor
from dinora.policy2 import extract_prob_from_policy


def softmax(x, tau=1.0):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


def dummy_wdl(scalar):
    wdl = softmax(np.array([scalar, 0, -scalar]))
    return wdl


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=filters),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.body(x) + x)


class AlphaNet(pl.LightningModule):
    def __init__(
        self,
        filters: int = 256,
        res_blocks: int = 19,
        policy_channels: int = 64,
        value_channels: int = 8,
        value_fc_hidden: int = 256,
        learning_rate: float = 0.001,
        lr_scheduler_gamma: float = 1.0,
        lr_scheduler_freq: int = 1000,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.lr_scheduler_freq = lr_scheduler_freq

        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels=18,
                out_channels=filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *(ResBlock(filters) for _ in range(res_blocks))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=filters,
                out_channels=policy_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=policy_channels * 8 * 8, out_features=1880),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=filters,
                out_channels=value_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=value_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=value_channels * 8 * 8,
                out_features=value_fc_hidden
            ),
            nn.ReLU(),
            nn.Linear(in_features=value_fc_hidden, out_features=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.convblock(x)
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x)
    
    def training_step(self, batch, batch_idx):
        x, (y_policy, y_value) = batch
        batch_len = len(x)

        y_hat_policy, y_hat_value = self(x)

        policy_loss = F.cross_entropy(y_hat_policy, y_policy)
        value_loss = F.mse_loss(y_hat_value, y_value)
        cumulative_loss = policy_loss + value_loss

        policy_accuracy = (
            (y_hat_policy.argmax(1) == y_policy)
            .float()
            .sum()
            .item()
        ) / batch_len

        self.log_dict({
            'train/policy_accuracy': policy_accuracy,
            'train/policy_loss': policy_loss,
            'train/value_loss': value_loss,
            'train/cumulative_loss': cumulative_loss
        })
        
        return cumulative_loss
    
    def validation_step(self, batch, batch_idx):
        x, (y_policy, y_value) = batch
        batch_len = len(x)
        y_hat_policy, y_hat_value = self(x)
        
        policy_accuracy = (
            (y_hat_policy.argmax(1) == y_policy)
            .float()
            .sum()
            .item()
        ) / batch_len

        self.log_dict({
            "validation/policy_accuracy": policy_accuracy,
            "validation/policy_loss": F.cross_entropy(y_hat_policy, y_policy).item(),
            "validation/value_loss": F.mse_loss(y_hat_value, y_value).item(),
        })

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(
            optimizer,
            step_size=1,
            gamma=self.lr_scheduler_gamma,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': self.lr_scheduler_freq,
            },
        }
    
    def eval_by_network(self, board: chess.Board):
        board_tensor = board_to_tensor(board)
        get_prob = extract_prob_from_policy

        with torch.no_grad():
            raw_policy, raw_value = self(
                torch.from_numpy(board_tensor).reshape((1, 18, 8, 8)).to(self.device)
            )

        outcome_logits = raw_value[0].cpu().item()
        outcomes_probs = dummy_wdl(outcome_logits)

        policy = raw_policy[0].cpu()

        moves = list(board.legal_moves)
        move_logits = [get_prob(policy, move, not board.turn) for move in moves]
        
        move_priors = softmax(np.array(move_logits))
        priors = dict(zip(moves, move_priors))

        return priors, outcomes_probs


if __name__ == '__main__':
    net = AlphaNet()

    x = torch.zeros((6, 18, 8, 8))
    policy, value = net(x)
    print(policy.shape, value.shape)
