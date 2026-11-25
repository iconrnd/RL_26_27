# AlphaGo Zero on a 5×5 Go board — starter repository

Starter repository of the course project of *Reinforcement Learning*
(Jagiellonian University, 2026/2027). The project is described in
`specification.pdf`; this repository fixes the files, the signatures, and the
public tests to which the specification refers.

## Getting started

1. Create a **private** copy of this repository (on GitHub: *Use this
   template → Create a new repository → Private*) and add the instructor as a
   collaborator (*Settings → Collaborators*).
2. Work through the three stages in order. `STAGE` in `az5/__init__.py` names
   the stage being worked on; the public tests of later stages are skipped.
3. Run the tests from the repository root:

   ```
   pip install -r requirements.txt    # not needed in Colab
   pytest
   ```

4. Submit a stage by setting `STAGE`, committing, and pushing the tag:

   ```
   git tag stage1
   git push origin stage1
   ```

   At the tag `stage<k>`, `STAGE` equals k and `pytest` passes.

## Layout

```
AGENTS.md, CLAUDE.md   instructions for coding agents        (provided)
specification.pdf      the project specification             (provided)
az5/
  __init__.py          STAGE
  config.py            all hyperparameters                   (provided, extendable)
  go.py                5x5 Go rules engine                   (stage 1)
  tictactoe.py         same game interface, for tests        (provided)
  data.py              encoding, symmetries, records, buffer (stage 1)
  agents.py            random, heuristic, human, MCTS agents (stages 1-3)
  arena.py             games, intervals, matches, gating     (stages 1-2)
  mcts.py              tree search, UCT and PUCT             (stages 2-3)
  net.py               policy-value network, evaluator       (stage 3)
  selfplay.py          self-play games                       (stage 3)
  train.py             loss, training, main loop             (stage 3)
tests/
  stage1/ stage2/ stage3/   public tests (provided)
  own/                      own tests
notebooks/             stage1.ipynb, stage2.ipynb, stage3.ipynb
reports/               stage1.pdf, stage2.pdf, stage3.pdf
runs/                  SGF files, training log, final weights
```

Every function to be written carries a docstring with its exact behaviour and
raises `NotImplementedError` until it is implemented.

## Colab or a local machine

The project needs no GPU. Every stage, including the Stage 3 training runs,
can be computed in Google Colab or on a local machine; the notebooks run
unchanged in both. Submissions are checked by running the notebooks in a fresh
Colab runtime, and each report names the machine that computed its results.

**Colab.** Each notebook starts with a setup cell that clones the repository
into the Colab runtime and pulls the latest commit. For a private repository,
create a GitHub personal access token with read access to it and store it as
the Colab secret `GITHUB_TOKEN` (key icon in the left bar); set `REPO_URL` in
the setup cell once. The Stage 3 notebook keeps its training runs on Google
Drive, so that they resume after a disconnect.

**Local machine.** With Python 3.10 or newer, from the repository root:

```
pip install -r requirements.txt
jupyter lab
```

The setup cell switches to the repository root, and the Stage 3 notebook keeps
its training runs in `work/`, which is not committed.

In both cases code changes are made in the repository and committed; the
notebooks only call the package.

## Coding agents

AI coding agents may be used. `AGENTS.md` (read by most coding agents) and its
copy `CLAUDE.md` state the code rules of the specification; keep them in the
repository unchanged.
