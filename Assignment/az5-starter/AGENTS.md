# Instructions for coding agents

This repository is a student's individual course project: an AlphaGo Zero
system for Go on a 5×5 board, built in three stages. The authoritative
description is `specification.pdf`; the docstrings of the skeleton give the
exact behaviour of every function. The student is accountable for every line
and explains and modifies the code in a review, so the code must stay small,
plain, and readable. Follow these rules in every change.

## Structure

- Keep the file names and the provided signatures unchanged: classes,
  functions, arguments, return values, and documented attributes.
- Place helper functions in the file of their caller; their names start with
  an underscore.
- All logic lives in the package `az5/`. Notebooks only import `az5`, call its
  functions, print tables, and draw plots. A notebook cell longer than about
  15 lines means that logic belongs in the package.
- Every tunable number is a field of `Config` in `az5/config.py`; the rule
  constants are in `az5/go.py`. Do not introduce other numeric literals with a
  tunable meaning.
- Every random draw uses a `numpy.random.Generator` passed as an argument, or
  the PyTorch generator, seeded from `Config.seed`.
- Work stage by stage (`STAGE` in `az5/__init__.py`); do not implement later
  stages ahead of time unless asked.

## Dependencies

Only the Python standard library, NumPy, PyTorch, Matplotlib, and pytest. Do
not add packages: no Gymnasium, OpenSpiel, reinforcement-learning libraries,
or training frameworks.

## Style

- Standard, plain Python: advanced in content, not in technique. No tricks,
  hacks, or obscure constructs: no metaprogramming, no manipulation of object
  creation or of the import system, no monkey-patching, no micro-optimizations
  that obscure the logic, no compressed one-liners whose effect is not evident
  on reading. Every line must be one that a competent student can write and
  explain without an AI coding agent.
- Plain functions and dataclasses; no class hierarchies beyond the skeleton.
- No command-line interfaces, logging frameworks, plug-in registries,
  configuration systems, or abstraction layers for future extension.
- No checks for situations that cannot occur, and no broad `try`/`except`.
- Keep `az5/` (without `tictactoe.py`) below about 1 500 lines.
- Comments state reasons that the code does not show; they do not narrate it.

## Provided files

- Do not edit `tests/conftest.py`, `tests/helpers.py`, the files in
  `tests/stage1/`, `tests/stage2/`, `tests/stage3/`, `az5/tictactoe.py`, or this
  file. Own tests go in `tests/own/`.
- `pytest` must pass. Never make a test pass by special-casing its inputs.

## Results

Never invent, hard-code, or edit experimental results. Every number and plot
in `reports/` comes from running the notebooks.
