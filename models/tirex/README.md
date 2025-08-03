---
datasets:
- autogluon/chronos_datasets
- Salesforce/GiftEvalPretrain
pipeline_tag: time-series-forecasting
library_name: tirex
license: other
license_link: https://huggingface.co/NX-AI/TiRex/blob/main/LICENSE
license_name: nx-ai-community-license
---

# TiRex

TiRex is a **time-series foundation model** designed for **time series forecasting**,
with the emphasis to provide state-of-the-art forecasts for both short- and long-term forecasting horizon.
TiRex is **35M parameter** small and is based on the **[xLSTM architecture](https://github.com/NX-AI/xlstm)** allowing fast and performant forecasts.
The model is described in the paper [TiRex: Zero-Shot Forecasting across Long and Short Horizons with Enhanced In-Context Learning](https://arxiv.org/abs/2505.23719).

### Key Facts:

- **Zero-Shot Forecasting**:
  TiRex performs forecasting without any training on your data. Just download and forecast.

- **Quantile Predictions**:
  TiRex not only provides point estimates but provides quantile estimates.

- **State-of-the-art Performance over Long and Short Horizons**:
  TiRex achieves top scores in various time series forecasting benchmarks, see [GiftEval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [ChronosZS](https://huggingface.co/spaces/autogluon/fev-leaderboard).
  These benchmark show that TiRex provides great performance for both long and short-term forecasting.

## Quick Start

The inference code is available on [GitHub](https://github.com/NX-AI/tirex).

### Installation

TiRex is currently only tested on *Linux systems* and Nvidia GPUs with compute capability >= 8.0.
If you want to use different systems, please check the [FAQ in the code repository](https://github.com/NX-AI/tirex?tab=readme-ov-file#faq--troubleshooting).
It's best to install TiRex in the specified conda environment.
The respective conda dependency file is [requirements_py26.yaml](https://github.com/NX-AI/tirex/blob/main/requirements_py26.yaml).

```sh
# 1) Setup and activate conda env from ./requirements_py26.yaml
git clone github.com/NX-AI/tirex
conda env create --file ./tirex/requirements_py26.yaml
conda activate tirex

# 2) [Mandatory] Install Tirex

## 2a) Install from source
git clone github.com/NX-AI/tirex  # if not already cloned before
cd tirex
pip install -e .

# 2b) Install from PyPi (will be available soon)

# 2) Optional: Install also optional dependencies
pip install .[gluonts]      # enable gluonTS in/output API
pip install .[hfdataset]    # enable HuggingFace datasets in/output API
pip install .[notebooks]    # To run the example notebooks
```

### Inference Example

```python
import torch
from tirex import load_model, ForecastModel

model: ForecastModel = load_model("NX-AI/TiRex")
data = torch.rand((5, 128))  # Sample Data (5 time series with length 128)
forecast = model.forecast(context=data, prediction_length=64)
```

We provide an extended quick start example in the [GitHub repository](https://github.com/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb).

### Troubleshooting / FAQ

If you have problems please check the FAQ / Troubleshooting section in the [GitHub repository](https://github.com/NX-AI/tirex)
and feel free to create a GitHub issue or start a discussion.


### Training Data

- [chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets) (Subset - Zero Shot Benchmark data is not used for training - details in the paper)
- [GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) (Subset - details in the paper)
- Synthetic Data

## Cite

If you use TiRex in your research, please cite our work: 

```bibtex
@article{auerTiRexZeroShotForecasting2025,
  title = {{{TiRex}}: {{Zero-Shot Forecasting Across Long}} and {{Short Horizons}} with {{Enhanced In-Context Learning}}},
  author = {Auer, Andreas and Podest, Patrick and Klotz, Daniel and B{\"o}ck, Sebastian and Klambauer, G{\"u}nter and Hochreiter, Sepp},
  journal = {ArXiv},
  volume = {2505.23719},   
  year = {2025}
}
```

## License

TiRex is licensed under the [NXAI community license](./LICENSE).