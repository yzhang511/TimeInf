### TimeInf: Time Series Data Contribution via Influence Functions

<p align="center">
    <img src=assets/figure.jpg />
</p>

### Environment setup

```
conda create --name timeinf
conda activate timeinf
git clone https://github.com/yzhang511/TimeInf.git
cd TimeInf
pip install -e .
```

### Datasets

Datasets for time series anomaly detection are available [here](https://drive.google.com/drive/folders/1VX2jmRdEvOM45U8ag62qL-qlVG0ieJ1A?usp=sharing). Please cite the original sources of the datasets.

### Models

Example scripts using TimeInf and other baselines for anomaly detection: `anomaly_detection/main_SMAP_MSL.py`.

Example notebooks for the application of linear, black-box, and non-parametric TimeInf: `demo`.
