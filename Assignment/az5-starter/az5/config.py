"""All hyperparameters of the project (specification, appendix "Default
configuration").

Every tunable number of the code is a field of Config; the rule constants
(board size, komi, move limit) are in go.py. Fields may be added when needed.
Variants are made with dataclasses.replace, for example
replace(Config(), selfplay_simulations=25).
"""

from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 0                      # seed of all random generators

    # Stage 2: matches and UCT
    opening_plies: int = 2             # random opening moves of a match pair
    c_uct: float = 1.4                 # exploration constant of UCT
    gate_pairs: int = 20               # pairs of games in the gate
    gate_threshold: float = 0.55       # win rate required to pass the gate
    baseline_simulations: int = 100    # simulations of the Stage 2 agent

    # Stage 3: network
    channels: int = 32                 # filters per convolution
    res_blocks: int = 4                # residual blocks

    # Stage 3: search
    c_puct: float = 1.5                # exploration constant of PUCT
    dirichlet_alpha: float = 0.5       # concentration of the root noise
    dirichlet_eps: float = 0.25        # weight of the root noise
    temp_moves: int = 8                # self-play moves played with tau = 1
    selfplay_simulations: int = 100    # simulations per self-play move
    eval_simulations: int = 100        # simulations per move in matches

    # Stage 3: training loop
    games_per_iter: int = 100          # self-play games per iteration
    iterations: int = 30               # iterations of the run
    buffer_capacity: int = 40_000      # positions kept in the replay buffer
    augment: bool = True               # symmetry augmentation when sampling
    batch_size: int = 256              # minibatch size
    train_steps: int = 200             # optimizer steps per iteration
    lr: float = 1e-3                   # Adam learning rate
    weight_decay: float = 1e-4         # L2 regularization
    use_gate: bool = True              # False: no gate (AlphaZero-style update)
    anchor_every: int = 5              # iterations between anchor matches
    anchor_pairs: int = 25             # pairs of games in an anchor match
