# Li-ion Cell Cycle-Life Prediction for Fast Cell Qualification

Cycling a Li-ion cell to end of life takes months. This project predicts a cell's cycle life
from its **first 100 cycles**, so a short-life cell can be flagged early, as an incoming
cell-qualification test instead of a full life test.

## Data

Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*,
Nature Energy (2019). 124 commercial LFP/graphite 18650 cells (A123 APR18650M1A) cycled under
72 fast-charging protocols, cycle life 150 to 2,300 cycles.
Download the three batch `.mat` files from https://data.matr.io/1/ into `data/`.

## Method

1. **Features from cycles 1 to 100**: variance and minimum of
   ΔQ(V) = Q₁₀₀(V) − Q₁₀(V) (discharge capacity vs. voltage), capacity-fade slope and
   intercept, capacity at cycle 2, internal resistance and its change, mean charge time.
2. **Models** on log₁₀(cycle life): a single-feature variance baseline, elastic net on all
   features, and gradient boosting.
3. **Qualification view**: recall on short-life cells (< 550 cycles) and the test-time
   reduction from stopping at cycle 100.

## Run

```bash
pip install -r requirements.txt
python cycle_life.py data/2017-05-12_batchdata.mat data/2017-06-30_batchdata.mat data/2018-04-12_batchdata.mat
```

Writes `metrics.json` (MAPE, RMSE, short-life recall per model).

## Limits

The cells are one chemistry and one format at 30 °C. Charge protocols outside the training
range, other chemistries (NMC) or other formats (4680) need new data before the model can be
trusted on a production line.
