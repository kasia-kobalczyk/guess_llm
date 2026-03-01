# Eliciting Numerical Predictive Distributions of LLMs Without Auto-Regression

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Paper](https://img.shields.io/badge/Paper-CC%20BY%204.0-orange)

<img src="https://github.com/kasia-kobalczyk/guess_llm/blob/master/assets/featured.png?raw=true" width="800"/>


## Overview
This repository contains the official implementation of the probing models presented in the ICLR 2026 paper [*Eliciting Numerical Predictive Distributions of LLMs Without Auto-Regression*](https://openreview.net/forum?id=jwzClhPT0j).
It provides code to:
- Prepare synthetic and real-world datasets
- Train probe models on LLM outputs
- Reproduce the main results and figures from the paper

**Hardware requirements:** GPU with 32GB+ VRAM recommended (required to compute LLM embeddings)

### Repository Structure

```
.
├── src/guess_llm/        # Core library code
├── bin/                  # Data generation scripts
├── notebooks/            # Figure and analysis notebooks
├── environment.yml       # Conda environment
└── README.md
```

## Step 0. Environment

Install and activate the conda environment with:

```
conda create -n guess_llm python=3.12 -y
conda activate guess_llm
pip install -e . 
```

## Step 1. Data Generation

To create the synthetic datasets used in sections 2 and 3 run 
```
bash bin/synthetic_data_prep.sh
```

To create the dataset based on the monash and darts datasets, run 
```
bash bin/real_data_prep.sh
```

Note: External datasets are subject to their original licenses and are not covered by this repository’s license.

By default, the datasets are generated with the `meta-llama/Llama-2-7b-hf` language model. This can be changed to other LLMs, e.g. `meta-llama/Meta-Llama-3-8B` or `microsoft/Phi-3.5-mini-instruct` by changing the `model_name` in the `.sh` scripts. 

## Step 2. Model Training

### Estimation of Scalar Statistics (mean, median, greedy; paper Section 2)

To train the magnitude-factorised regression models run:
```
python -m guess_llm.probe.train --config-name=multi_func_<SCALE> model=mag_reg ++model.target=<TARGET>
```
where `<SCALE>` should be set to one of `1.0`, `10.0`, `1000.0`, `10000.0`
and `<TARGET>` set to one of `greedy`, `mean`, `median`.

To specify the LLM used (if other than `Llama-2-7b-hf`) add the flag `++dataset.model_name=Meta-Llama-3-8B`.

The corresponding model will be saved under `./saves/` directory. To evaluate a specific model saved under `./saves/<SAVE_PATH>/best_model.ckpt` run:
```
python -m guess_llm.probe.test --save_dir=<SAVE_PATH>
```

### Quantile Regression (paper Section 3)

To train quantile regression models on the synthetic dataset with varying scale run:

```
python -m guess_llm.probe.train --config-name=multi_func_<SCALE> model=quantile_mag_reg
```

To train quantile regression models on real data run:

- Real (all)

```
python -m guess_llm.probe.train --config-name=monash_darts_combined ++dataset.split_file=random_splits_filtered.csv model=quantile_mag_reg
```

- Real (5 fold)

```
for k in 0 1 2 3 4
do
    python -m guess_llm.probe.train --config-name=monash_darts_combined ++dataset.split_file=kfold_splits_filtered_${k}.csv model=quantile_mag_reg
done
```

- Synth

```
python -m guess_llm.probe.train --config-name=multi_func_combined ++dataset.split_file=strong_filter.csv model=quantile_mag_reg
```

To evaluate the model trained on synthetic data on the Real (all) test set run:
```
python -m guess_llm.probe.test --save_dir=<SAVE_PATH> --test_dataset_name=monash_darts_combined --test_split_file=random_splits_filtered
```

## Step 3. Results Analysis and Visualisation

There are three notebooks in which results for different model / data types can be visualised:
- `visualise_center_predictions.ipynb`for the regression models with scalar targets
- `visualise_synthetic_uncertainty.ipynb` for the quantile regression models on synthetic data
- `visualise_real_data.ipynb` for the quantile regression models on real data

---

## Citation

```bibtex
@inproceedings{
piskorz2026eliciting,
title={Eliciting Numerical Predictive Distributions of {LLM}s Without Auto-Regression},
author={Julianna Piskorz and Katarzyna Kobalczyk and Mihaela van der Schaar},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=SsuBd46twl}
}
```
