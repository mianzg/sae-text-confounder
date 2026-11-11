# sae-text-confounder


Code for *Exploring Sparse Autoencoders in Text-Based Causal Confounding Adjustment* (EMNLP 2026) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2507/) · [arXiv](https://arxiv.org/abs/2609.01322)

- **Use it on your own data** → [Quickstart](#use-it-on-your-own-data)
- **Reproduce the paper** → [Reproduce](#reproduce-the-paper)

## Install

```bash
pip install -e .
```

## Use it on your own data

Your table needs a text column, a binary treatment column, a numeric outcome column.

```python
import pandas as pd
from causael import estimate_effect

df = pd.read_csv("your_data.csv")   # a text column "text", a binary treatment column "T", a numeric outcome column "Y"

result = estimate_effect(
    df,
    text="text",
    treatment="T",
    outcome="Y",
)
print(result)
```
For illustration, an example output is below. 
```
Effect  (method=matching, n=4182)

  Adjusted        +0.420    95% CI [0.180, 0.660]
  Unadjusted      +1.070    95% CI [0.910, 1.230]

Text confounders found and adjusted for  (4 of 128 SAE features):
  #37  prevalence 22%   SMD 0.61 -> 0.04
  #9   prevalence 14%   SMD 0.44 -> 0.07
  #88  prevalence 31%   SMD 0.28 -> 0.02
  #102 prevalence 9%   SMD 0.35 -> 0.11

Warning: feature #102 is still imbalanced after matching (SMD 0.11 > 0.10).
```

`result.confounders` is the table above as a DataFrame (feature id, prevalence, LRT p-value, SMD
before/after matching) and `result.save("out/")` writes it plus a summary CSV to disk.

## Reproduce the paper
Work-in-progress

## Citation & license

```bibtex
@inproceedings{zhong-etal-2026-exploring,
    title = "Exploring Sparse Autoencoders in Text-Based Causal Confounding Adjustment",
    author = "Zhong, Mian  and
      Keith, Katherine A.  and
      Field, Anjalie",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2507/",
    doi = "10.18653/2026.emnlp-main.2507",
    pages = "56985--57009",
    ISBN = "979-8-89176-459-0",
}
```

Code is MIT licensed ([LICENSE](LICENSE)). 