# CMI-RNN

Cell-type-specific Microcircuit Interaction RNN: a data-constrained recurrent
neural network that infers directed effective interactions between neuronal
types from transition-aligned calcium activity.

## Overview

The model consists of four independent LSTM subnetworks, one for each neuronal
type (**CaMKIIa, PV, SST, VIP**), coupled through a trainable 4x4 interaction
matrix (`synweights`). At every time step, each subnetwork receives a brief
shared input pulse plus the weighted previous outputs of the other subnetworks.
Each subnetwork has a single-unit dense readout, so the network predicts one
population-averaged trace per neuronal type.

The interaction matrix is indexed as `synweights[post, pre]`: rows are
postsynaptic and columns are presynaptic, so a positive value denotes an
excitatory effective influence and a negative value an inhibitory one. The
`VIP -> CaMKIIa` connection is fixed at zero as a structural constraint.
Fitted weights are **effective (phenomenological) interactions** consistent with
the observed dynamics; they do not establish underlying synaptic connectivity.

Training minimizes the mean squared error between predicted and observed
population-averaged traces. Because the fitted weights can depend on
initialization, the model is trained from many independent random
initializations; downstream analysis retains runs that fit the data well (for
example, `R2 >= 0.7`) and averages the weight matrices across retained runs.

## Data format

`cmi_rnn.py` expects a single NumPy array saved as `.npy` with shape
`(4, T, N)`:

- axis 0: neuronal type, in the order CaMKIIa, PV, SST, VIP
- axis 1: time frames per trial (`T`, default 200)
- axis 2: samples, e.g. individual neurons or transition-aligned events (`N`)

Traces should be z-scored and aligned to a behavioral transition onset. Each
trial has a brief input pulse applied to all four types; its window is set by
`STIM_ON` and `STIM_OFF` in the config.

Example layout:

```
data/
  transition_aligned_traces.npy   # shape (4, 200, N)
```

The example data are not included in this repository.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy

```bash
pip install numpy tensorflow
```

## Usage

1. Set `DATA_PATH` (and optionally `OUTPUT_DIR` and the training constants) at
   the top of `cmi_rnn.py`.
2. Run:

```bash
python cmi_rnn.py
```

The script trains `N_RUNS` models with independent initializations. Key
constants: `UNITS` (LSTM hidden size), `TOTAL_STEPS` (gradient updates per run),
`LR`, and `PRINT_EVERY`.

## Outputs

All files are written to `OUTPUT_DIR` (default `results/`) and use a 0-based run
index `i`:

| File | Content |
| --- | --- |
| `i_output1.npy` ... `i_output4.npy` | predicted trace for each neuronal type, shape `(T,)` |
| `i_observation1.npy` ... `i_observation4.npy` | observed population-averaged trace, shape `(T,)` |
| `i_weight.npy` | inferred 4x4 interaction matrix, indexed `[post, pre]` |

## Notes

- Mixed precision (`mixed_float16`) is enabled; the optimizer is wrapped in
  `LossScaleOptimizer`, and `cmi_rnn.py` handles both legacy and current
  TensorFlow loss-scaling APIs.
- Outputs are saved per run with `float32` precision.
- The training step is compiled with `tf.function`; network weights and
  optimizer state are reset between runs rather than rebuilding the graph.
- The MATLAB script `modeling_plot_batch_weightR2.m` reads these outputs for
  quality control (R2 filtering) and weight aggregation.

## Citation

Manuscript in preparation. Citation information will be added here.

## Contact

For questions, please open an issue or contact the corresponding authors.
