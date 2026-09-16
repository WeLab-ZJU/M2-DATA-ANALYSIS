"""Cell-type-specific Microcircuit Interaction RNN (CMI-RNN).
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ---------------------------------------------------------------------- config
DATA_PATH = os.path.join("data", "transition_aligned_traces.npy")  # (4, T, N)
OUTPUT_DIR = "results"
N_RUNS = 100          # independent random initializations
UNITS = 100           # hidden units per LSTM subnetwork
TIME_STEPS = 200      # frames per trial
TOTAL_STEPS = 1000    # gradient updates per run
LR = 8e-4
PRINT_EVERY = 50
STIM_ON, STIM_OFF = 20, 30    # input pulse window (frames)
FIXED_ZERO = (0, 3)           # VIP -> CaMKIIa connection, fixed at zero

# -------------------------------------------------------------------- GPU setup
print("GPU:", tf.config.list_physical_devices("GPU"))
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]], "GPU")

# Mixed precision: layers compute in float16 while variables remain float32.
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)
print("Compute dtype:", policy.compute_dtype, "| Variable dtype:", policy.variable_dtype)


class InteractiveRNN(tf.keras.Model):
    """Four LSTM subnetworks coupled by a trainable 4x4 interaction matrix."""

    def __init__(self, units):
        super().__init__()
        self.units = units

        self.rnn1 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.rnn2 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.rnn3 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.rnn4 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)

        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(1)
        self.dense3 = tf.keras.layers.Dense(1)
        self.dense4 = tf.keras.layers.Dense(1)

        # Interaction matrix, rows = postsynaptic, columns = presynaptic.
        initial_weights = np.full((4, 4), -0.5, dtype=np.float32)
        initial_weights[:, 0] = 0.5
        initial_weights[FIXED_ZERO] = 0.0
        np.fill_diagonal(initial_weights, 0.0)
        self.synweights = tf.Variable(initial_weights, trainable=True, name="synweights")

    def call(self, inputs, initial_states=None):
        input1, input2, input3, input4 = inputs

        # Keras casts model inputs to float16; keep interaction terms in float32.
        input1 = tf.cast(input1, tf.float32)
        input2 = tf.cast(input2, tf.float32)
        input3 = tf.cast(input3, tf.float32)
        input4 = tf.cast(input4, tf.float32)

        batch_size = tf.shape(input1)[0]

        # LSTM states are not autocasted, so initialize them in the compute dtype.
        state_dtype = self.compute_dtype
        if initial_states is None:
            state1 = [tf.zeros((batch_size, self.units), dtype=state_dtype),
                      tf.zeros((batch_size, self.units), dtype=state_dtype)]
            state2 = [tf.zeros((batch_size, self.units), dtype=state_dtype),
                      tf.zeros((batch_size, self.units), dtype=state_dtype)]
            state3 = [tf.zeros((batch_size, self.units), dtype=state_dtype),
                      tf.zeros((batch_size, self.units), dtype=state_dtype)]
            state4 = [tf.zeros((batch_size, self.units), dtype=state_dtype),
                      tf.zeros((batch_size, self.units), dtype=state_dtype)]
        else:
            state1, state2, state3, state4 = initial_states

        prev_output1 = tf.zeros((batch_size, 1))
        prev_output2 = tf.zeros((batch_size, 1))
        prev_output3 = tf.zeros((batch_size, 1))
        prev_output4 = tf.zeros((batch_size, 1))

        all_outputs1, all_outputs2, all_outputs3, all_outputs4 = [], [], [], []

        for t in range(input1.shape[1]):
            prev_state1, prev_state2 = state1, state2
            prev_state3, prev_state4 = state3, state4

            # Dense outputs are float16; cast back so the matrix product is float32.
            prev_output1 = tf.cast(prev_output1, tf.float32)
            prev_output2 = tf.cast(prev_output2, tf.float32)
            prev_output3 = tf.cast(prev_output3, tf.float32)
            prev_output4 = tf.cast(prev_output4, tf.float32)

            x1_t = input1[:, t, :]
            x2_t = input2[:, t, :]
            x3_t = input3[:, t, :]
            x4_t = input4[:, t, :]

            x1_t = x1_t + self.synweights[0, 1] * prev_output2 \
                        + self.synweights[0, 2] * prev_output3
            output1, h1, c1 = self.rnn1(tf.expand_dims(x1_t, 1), initial_state=prev_state1)
            state1 = [h1, c1]

            x2_t = x2_t + self.synweights[1, 0] * prev_output1 \
                        + self.synweights[1, 2] * prev_output3 \
                        + self.synweights[1, 3] * prev_output4
            output2, h2, c2 = self.rnn2(tf.expand_dims(x2_t, 1), initial_state=prev_state2)
            state2 = [h2, c2]

            x3_t = x3_t + self.synweights[2, 0] * prev_output1 \
                        + self.synweights[2, 1] * prev_output2 \
                        + self.synweights[2, 3] * prev_output4
            output3, h3, c3 = self.rnn3(tf.expand_dims(x3_t, 1), initial_state=prev_state3)
            state3 = [h3, c3]

            x4_t = x4_t + self.synweights[3, 0] * prev_output1 \
                        + self.synweights[3, 1] * prev_output2 \
                        + self.synweights[3, 2] * prev_output3
            output4, h4, c4 = self.rnn4(tf.expand_dims(x4_t, 1), initial_state=prev_state4)
            state4 = [h4, c4]

            output1 = self.dense1(output1)
            output2 = self.dense2(output2)
            output3 = self.dense3(output3)
            output4 = self.dense4(output4)

            prev_output1 = output1[:, 0, :]
            prev_output2 = output2[:, 0, :]
            prev_output3 = output3[:, 0, :]
            prev_output4 = output4[:, 0, :]

            all_outputs1.append(prev_output1)
            all_outputs2.append(prev_output2)
            all_outputs3.append(prev_output3)
            all_outputs4.append(prev_output4)

        outputs1 = tf.stack(all_outputs1, axis=1)   # (batch, time, 1)
        outputs2 = tf.stack(all_outputs2, axis=1)
        outputs3 = tf.stack(all_outputs3, axis=1)
        outputs4 = tf.stack(all_outputs4, axis=1)

        return (outputs1, outputs2, outputs3, outputs4), (state1, state2, state3, state4)


# ---------------------------------------------------------------------- data
all_data = np.load(DATA_PATH)
n_samples = all_data.shape[2]
print("data:", all_data.shape)

ca_1_exp, ca_2_exp = all_data[0], all_data[1]
ca_3_exp, ca_4_exp = all_data[2], all_data[3]

target1 = ca_1_exp.reshape(1, TIME_STEPS, n_samples)
target2 = ca_2_exp.reshape(1, TIME_STEPS, n_samples)
target3 = ca_3_exp.reshape(1, TIME_STEPS, n_samples)
target4 = ca_4_exp.reshape(1, TIME_STEPS, n_samples)

# Identical brief pulse delivered to all four subnetworks.
input1 = np.zeros((1, TIME_STEPS, 1))
input2 = np.zeros((1, TIME_STEPS, 1))
input3 = np.zeros((1, TIME_STEPS, 1))
input4 = np.zeros((1, TIME_STEPS, 1))
for x in (input1, input2, input3, input4):
    x[0, STIM_ON:STIM_OFF, 0] = 1

model = InteractiveRNN(UNITS)
# Loss scaling is required for float16 gradients and must wrap the optimizer.
optimizer = mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.Adam(LR, clipnorm=1.0)
)
loss_fn = tf.keras.losses.MeanSquaredError()

inputs = tuple(tf.convert_to_tensor(x, dtype=tf.float32) for x in (input1, input2, input3, input4))
targets = tuple(tf.convert_to_tensor(t, dtype=tf.float32) for t in (target1, target2, target3, target4))
input1, input2, input3, input4 = inputs
target1, target2, target3, target4 = targets

# Build variables once; later runs reuse the same graph and only reset values.
_ = model(inputs)

# Reset specifications: Keras default initializers, redrawn for every run.
reset_specs = []
for rnn in (model.rnn1, model.rnn2, model.rnn3, model.rnn4):
    reset_specs.append((rnn.cell.kernel, tf.keras.initializers.GlorotUniform()))
    reset_specs.append((rnn.cell.recurrent_kernel, tf.keras.initializers.Orthogonal()))
    reset_specs.append((rnn.cell.bias, tf.keras.initializers.Zeros()))
for dense in (model.dense1, model.dense2, model.dense3, model.dense4):
    reset_specs.append((dense.kernel, tf.keras.initializers.GlorotUniform()))
    reset_specs.append((dense.bias, tf.keras.initializers.Zeros()))

synweights_init = np.full((4, 4), -0.5, dtype=np.float32)
synweights_init[:, 0] = 0.5
synweights_init[FIXED_ZERO] = 0.0
np.fill_diagonal(synweights_init, 0.0)


def reset_experiment():
    """Redraw all weights and clear optimizer state without rebuilding the graph."""
    for variable, initializer in reset_specs:
        variable.assign(initializer(variable.shape, dtype=variable.dtype))
    model.synweights.assign(synweights_init)

    # Adam slots and dynamic loss-scale state; variable names differ across TF versions.
    for variable in optimizer.variables():
        name = variable.name.lower()
        if "loss_scale" in name:
            variable.assign(2 ** 15)
        elif "beta" in name:
            variable.assign(tf.ones_like(variable))
        else:
            variable.assign(tf.zeros_like(variable))


@tf.function
def train_step():
    with tf.GradientTape() as tape:
        (outputs1, outputs2, outputs3, outputs4), _ = model(inputs)
        # Cast outputs to float32 before the loss to keep the reduction precise.
        loss = (loss_fn(target1, tf.cast(outputs1, tf.float32)) +
                loss_fn(target2, tf.cast(outputs2, tf.float32)) +
                loss_fn(target3, tf.cast(outputs3, tf.float32)) +
                loss_fn(target4, tf.cast(outputs4, tf.float32)))
        # Legacy LossScaleOptimizer scales manually, and only inside the tape.
        scaled_loss = optimizer.get_scaled_loss(loss) if hasattr(optimizer, "get_scaled_loss") else loss
    grads = tape.gradient(scaled_loss, model.trainable_variables)
    if hasattr(optimizer, "get_scaled_loss"):
        grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ------------------------------------------------------------------- training
os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx in range(N_RUNS):
    reset_experiment()

    t_start = time.time()
    for step in range(TOTAL_STEPS):
        loss = train_step()
        if step % PRINT_EVERY == 0 or step == TOTAL_STEPS - 1:
            print(f"run {idx} step {step}/{TOTAL_STEPS} loss {loss.numpy():.6f}")
    print(f"run {idx} training time {time.time() - t_start:.1f}s")

    (test_output1, test_output2, test_output3, test_output4), _ = model(inputs)
    print("interaction weights:\n", model.synweights.numpy())

    for i, out in enumerate((test_output1, test_output2, test_output3, test_output4), start=1):
        np.save(os.path.join(OUTPUT_DIR, f"{idx}_output{i}.npy"),
                tf.cast(out, tf.float32).numpy()[0, :, 0])
    for i, obs in enumerate((ca_1_exp, ca_2_exp, ca_3_exp, ca_4_exp), start=1):
        np.save(os.path.join(OUTPUT_DIR, f"{idx}_observation{i}.npy"), np.mean(obs, axis=1))
    np.save(os.path.join(OUTPUT_DIR, f"{idx}_weight.npy"), model.synweights.numpy())
