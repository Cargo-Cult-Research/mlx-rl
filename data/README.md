# data/

## mbpp_sanitized.json

The **sanitized MBPP** split (427 crowd-sourced Python programming problems
with unit-test asserts), used by the `code` task
(`src/mlx_rl/tasks/code.py`).

- **Source:** MBPP — *Mostly Basic Python Problems* — released by Google
  Research with the paper *Program Synthesis with Large Language Models*
  (Austin et al., 2021, [arXiv:2108.07732](https://arxiv.org/abs/2108.07732));
  dataset files at
  [github.com/google-research/google-research/tree/master/mbpp](https://github.com/google-research/google-research/tree/master/mbpp).
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Redistributed here unmodified (JSON re-serialization only). The repository's
  MIT license applies to the mlx-rl code, **not** to this dataset.

```bibtex
@article{austin2021program,
  title   = {Program Synthesis with Large Language Models},
  author  = {Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma,
             Maarten and Michalewski, Henryk and Dohan, David and Jiang,
             Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and
             Sutton, Charles},
  journal = {arXiv preprint arXiv:2108.07732},
  year    = {2021}
}
```

(The `math` task's dataset — agentica-org/DeepScaleR-Preview-Dataset, MIT —
is *not* redistributed; it is fetched from the Hugging Face Hub at first use.)

## arxiv_snapshot.jsonl

Frozen arXiv **metadata** (id, title, authors, first-version date,
categories) used by the `qa_arxiv` task as its search index, so training
never hits the live arXiv API. Built by `scripts/fetch_arxiv_snapshot.py`
on 2026-08-16: 97 hand-listed well-known papers, the first 40 cs.LG
submissions of every month 2023-01..2026-08, and 300 `fictional_*` titles
recombined from real title halves (no authors/date; searching for them must
come back empty). arXiv metadata is released under
[CC0 1.0](https://info.arxiv.org/help/api/tou.html); no abstracts or
full text are included.
