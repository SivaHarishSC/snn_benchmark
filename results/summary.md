| Model | T | Test acc | Spikes/neuron/inference | Ops/inference | Energy est. (µJ, theoretical) | CNN/SNN energy ratio | GPU latency (ms / batch 256) |
|---|---|---|---|---|---|---|---|
| CNN | – | 0.9915 ± 0.00038 | – | MACs 1.534e+06 ± 0 | 7.056 ± 0 | 1.00 | 0.8453 ± 0.37 |
| SNN | 5 | 0.9841 ± 0.003 | 0.6447 ± 0.033 | SOPs 1.347e+06 ± 1.1e+05 | 1.212 ± 0.097 | 5.82× | 14.27 ± 4.1 |
| SNN | 10 | 0.9888 ± 0.0017 | 1.021 ± 0.12 | SOPs 2.169e+06 ± 3.1e+05 | 1.952 ± 0.28 | 3.61× | 33.04 ± 2.8 |
| SNN | 25 | 0.9892 ± 0.0023 | 1.828 ± 0.15 | SOPs 4.282e+06 ± 4.5e+05 | 3.854 ± 0.4 | 1.83× | 73.48 ± 33 |

FGSM accuracy (SNN at T=10), mean ± std over seeds:

| Model | ε=0.0 | ε=0.05 | ε=0.1 | ε=0.2 |
|---|---|---|---|---|
| CNN | 0.9915 ± 0.00038 | 0.9534 ± 0.0059 | 0.8176 ± 0.032 | 0.3614 ± 0.043 |
| SNN T=10 | 0.9888 ± 0.0017 | 0.9705 ± 0.0037 | 0.9282 ± 0.0039 | 0.7394 ± 0.048 |

Energy: Theoretical estimate for neuromorphic hardware (Horowitz 2014, 45 nm). Not measured; a GPU saves no energy running an SNN.
Epochs per run: 10. Runs with thermal throttle flag during latency: 0/12.
Runtime of this invocation: 16.9 min (runs resumed from MLflow: 10). Sum of training time over all 12 runs: 57.0 min. GPU/driver: NVIDIA GeForce RTX 4050 Laptop GPU, 566.07