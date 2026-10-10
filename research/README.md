# Research

Experiments that confirm or refute claims about `aethelnet-core`. Each notebook states the claim, the test and
the verdict, and is meant to be re-run when the code changes.

| Notebook | What it checks |
|---|---|
| [`lgnn_validation.ipynb`](lgnn_validation.ipynb) | `LiquidGraph` dynamics, learned modules, memory scaling, sharding, the edge rule, task value vs. baselines, and the tether |

The `.py` file next to each notebook is the source of truth ([Jupytext](https://jupytext.readthedocs.io) percent
format, readable diffs); the `.ipynb` is generated from it and committed with outputs so results render on GitHub.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" -r research/requirements.txt
cd research
jupytext --to ipynb --update lgnn_validation.py
jupyter nbconvert --to notebook --execute --inplace lgnn_validation.ipynb   # ~10 min on a laptop CPU
```

When you change `LiquidGraph`, re-run the notebook and update the verdicts in its summary table.
