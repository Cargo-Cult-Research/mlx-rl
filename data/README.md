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
