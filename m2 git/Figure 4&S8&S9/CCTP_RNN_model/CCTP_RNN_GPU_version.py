
import time

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Check GPU availability.
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth.
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    gpu0 = gpus[0]  # Use only GPU 0 if multiple GPUs are available.
    tf.config.experimental.set_memory_growth(gpu0, True)  # Allocate GPU memory on demand.
    tf.config.set_visible_devices([gpu0], "GPU")

# Enable mixed-precision training.
from tensorflow.keras import mixed_precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype:', policy.compute_dtype)
print('Variable dtype:', policy.variable_dtype)

class InteractiveRNN(tf.keras.Model):
    def __init__(self, units, input_dim):
        super(InteractiveRNN, self).__init__()
        self.units = units
        self.input_dim = input_dim

        # Define four RNN cores (return_sequences=True).
        self.rnn1 = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True
        )
        self.rnn2 = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True
        )
        self.rnn3 = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True
        )
        self.rnn4 = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            return_state=True
        )

        # Add Dense layers to compress each output to one dimension.
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(1)
        self.dense3 = tf.keras.layers.Dense(1)
        self.dense4 = tf.keras.layers.Dense(1)

        # Define a trainable 4x4 scalar weight matrix.
        # Initialized to 0.5; the diagonal is zero (no self-connections).
        initial_weights = np.full((4, 4), -0.5)
        initial_weights[:, 0] = 0.5
        initial_weights[0, 3] = 0
        np.fill_diagonal(initial_weights, 0.0)

        # Convert to a tensor.
        initial_weights = tf.constant(initial_weights, dtype=tf.float32)

        # Define the trainable weight matrix.
        # Interaction weights: from_weight -> to_weight
        # rows = to_weight, columns = from_weight
        # Positive = excitatory; negative = inhibitory.
        # VIP -> CaMKII is structurally zero (the term is omitted in RNN1).
        self.synweights = tf.Variable(initial_weights, trainable=True, name='weights')

    def call(self, inputs, initial_states=None):
        # Unpack inputs.
        input1, input2, input3, input4 = inputs
        # Under mixed precision, Keras automatically casts float32 inputs to
        # float16 at the model boundary, while the interaction terms
        # (interaction weights x prev_output) are computed in float32. Cast them back to
        # float32 here so the addition has a consistent dtype.
        # (No-op in float32 mode; equivalent to v17.)
        input1 = tf.cast(input1, tf.float32)
        input2 = tf.cast(input2, tf.float32)
        input3 = tf.cast(input3, tf.float32)
        input4 = tf.cast(input4, tf.float32)

        # Get the batch size dynamically.
        batch_size = tf.shape(input1)[0]

        # Initialize the states of the four RNNs.
        # Under mixed precision, LSTM initial_state is not automatically cast
        # by Keras; create it with the compute dtype (float16) to avoid dtype
        # mismatches in the internal matmul.
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

        # Initialize the first prev_output.
        prev_output1 = tf.zeros((tf.shape(input1)[0], 1))
        prev_output2 = tf.zeros((tf.shape(input1)[0], 1))
        prev_output3 = tf.zeros((tf.shape(input1)[0], 1))
        prev_output4 = tf.zeros((tf.shape(input1)[0], 1))

        # Store outputs for all time steps.
        all_outputs1, all_outputs2, all_outputs3, all_outputs4 = [], [], [], []

        # Process time step by time step.
        for t in range(input1.shape[1]):
            # Save the previous states.
            prev_state1 = state1
            prev_state2 = state2
            prev_state3 = state3
            prev_state4 = state4

            # Under mixed precision, Dense outputs are float16 while the
            # interaction weights are float32 variables (tf.Variable is not
            # autocast by Keras). TensorFlow does not perform implicit type
            # promotion, so the interaction terms must be computed in float32.
            prev_output1 = tf.cast(prev_output1, tf.float32)
            prev_output2 = tf.cast(prev_output2, tf.float32)
            prev_output3 = tf.cast(prev_output3, tf.float32)
            prev_output4 = tf.cast(prev_output4, tf.float32)

            # Get the input at the current time step.
            x1_t = input1[:, t, :]
            x2_t = input2[:, t, :]
            x3_t = input3[:, t, :]
            x4_t = input4[:, t, :]

            # RNN1 (receives weighted outputs from the other RNNs).
            x1_t = x1_t + \
                        self.synweights[0, 1] * prev_output2 + \
                        self.synweights[0, 2] * prev_output3
            output1, new_state1_h, new_state1_c = self.rnn1(
                tf.expand_dims(x1_t, 1),
                initial_state=prev_state1
            )
            state1 = [new_state1_h, new_state1_c]
            # RNN2 (receives weighted outputs from the other RNNs).
            x2_t = x2_t + \
                           self.synweights[1, 0] * prev_output1 + \
                           self.synweights[1, 2] * prev_output3 + \
                           self.synweights[1, 3] * prev_output4
            output2, new_state2_h, new_state2_c  = self.rnn2(
                tf.expand_dims(x2_t, 1),
                initial_state=prev_state2
            )
            state2 = [new_state2_h, new_state2_c]
            # RNN3 (receives weighted outputs from the other RNNs).
            x3_t = x3_t + \
                           self.synweights[2, 0] * prev_output1 + \
                           self.synweights[2, 1] * prev_output2 + \
                           self.synweights[2, 3] * prev_output4
            output3, new_state3_h, new_state3_c = self.rnn3(
                tf.expand_dims(x3_t, 1),
                initial_state=prev_state3
            )
            state3 = [new_state3_h, new_state3_c]
            # RNN4 (receives weighted outputs from the other RNNs).
            x4_t = x4_t + \
                           self.synweights[3, 0] * prev_output1 + \
                           self.synweights[3, 1] * prev_output2 + \
                           self.synweights[3, 2] * prev_output3

            output4, new_state4_h, new_state4_c = self.rnn4(
                tf.expand_dims(x4_t, 1),
                initial_state=prev_state4
            )
            state4 = [new_state4_h, new_state4_c]
            # Compress outputs to one dimension with Dense layers.
            output1 = self.dense1(output1)
            output2 = self.dense2(output2)
            output3 = self.dense3(output3)
            output4 = self.dense4(output4)

            prev_output1, prev_output2, prev_output3, prev_output4 = output1[:, 0, :], output2[:, 0, :], output3[:, 0, :], output4[:, 0, :]
            # Collect outputs (remove the time-step dimension).
            all_outputs1.append(output1[:, 0, :])
            all_outputs2.append(output2[:, 0, :])
            all_outputs3.append(output3[:, 0, :])
            all_outputs4.append(output4[:, 0, :])

        # Stack outputs over time.
        outputs1 = tf.stack(all_outputs1, axis=1)  # shape: (batch, time, 1)
        outputs2 = tf.stack(all_outputs2, axis=1)
        outputs3 = tf.stack(all_outputs3, axis=1)
        outputs4 = tf.stack(all_outputs4, axis=1)

        # Return format: (outputs1, outputs2, outputs3, outputs4), (state1, state2, state3, state4)
        return (outputs1, outputs2, outputs3, outputs4), (state1, state2, state3, state4)

# Parameters.
input_dim = 1
units = 100
time_steps = 200  # 200 frames at 20 Hz: -5 to 5 s relative to onset
total_steps = 1000   # 1,000 gradient updates per run
print_every = 50    # Loss printing interval

DATA_PATH = ""
all_data = np.load(DATA_PATH)
print(all_data.shape)
sample_num = all_data.shape[2]



# Read data.
data = all_data

ca_1_exp = data[0,:,:]
ca_2_exp = data[1,:,:]
ca_3_exp = data[2,:,:]
ca_4_exp = data[3,:,:]
target1 = ca_1_exp.reshape(1, time_steps, sample_num )
target2 = ca_2_exp.reshape(1, time_steps, sample_num )
target3 = ca_3_exp.reshape(1, time_steps, sample_num )
target4 = ca_4_exp.reshape(1, time_steps, sample_num )


# Input data.
t = np.linspace(0, 10, time_steps)

input1 = np.zeros((1, time_steps, 1))
input1[0, 20:30, 0] = 1

input2 = np.zeros((1, time_steps, 1))
input2[0, 20:30, 0] = 1

input3 = np.zeros((1, time_steps, 1))
input3[0, 20:30, 0] = 1

input4 = np.zeros((1, time_steps, 1))
input4[0, 20:30, 0] = 1


model = InteractiveRNN(units, input_dim)
# Under mixed precision, the optimizer must be wrapped in LossScaleOptimizer;
# otherwise float16 gradients underflow (Adam does not handle loss scaling
# automatically).
optimizer = mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.Adam(0.0008, clipnorm=1.0)
)
loss_fn = tf.keras.losses.MeanSquaredError()
# Convert input data to tensors.
input1 = tf.convert_to_tensor(input1, dtype=tf.float32)
input2 = tf.convert_to_tensor(input2, dtype=tf.float32)
input3 = tf.convert_to_tensor(input3, dtype=tf.float32)
input4 = tf.convert_to_tensor(input4, dtype=tf.float32)
# Convert target data to tensors.
target1 = tf.convert_to_tensor(target1, dtype=tf.float32)
target2 = tf.convert_to_tensor(target2, dtype=tf.float32)
target3 = tf.convert_to_tensor(target3, dtype=tf.float32)
target4 = tf.convert_to_tensor(target4, dtype=tf.float32)

inputs = (input1, input2, input3, input4)
targets = (target1, target2, target3, target4)

# Run one forward pass first to create the LSTM/Dense variables lazily.
# Subsequent runs reuse the same variables and graph; only their values are reset.
_ = model(inputs)

# Collect random initializers matching the Keras defaults.
# Each call to an initializer draws new random values, preserving the
# independent-initialization semantics of each run.
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
synweights_init[0, 3] = 0.0
np.fill_diagonal(synweights_init, 0.0)


def reset_experiment():
    """Reset model and optimizer state without rebuilding or retracing them."""
    for variable, initializer in reset_specs:
        variable.assign(initializer(variable.shape, dtype=variable.dtype))

    model.synweights.assign(synweights_init)

    # Reset Adam slots, iteration count, and dynamic loss-scale state.
    # The exact optimizer variable names differ slightly between TF versions.
    for variable in optimizer.variables():
        variable_name = variable.name.lower()
        if 'loss_scale' in variable_name:
            variable.assign(2 ** 15)
        elif 'beta' in variable_name:
            variable.assign(tf.ones_like(variable))
        else:
            variable.assign(tf.zeros_like(variable))


# Training step: compiled into a graph; forward pass, loss, gradients, and
# update are performed in one call.
# Note: the interaction weight matrix is a Model attribute variable and is
# automatically included in trainable_variables.
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        (outputs1, outputs2, outputs3, outputs4), _ = model(inputs)
        # Under mixed precision, model outputs are float16; cast back to
        # float32 for the loss (MSE does not unify dtypes automatically, and
        # float32 summation avoids precision loss).
        loss = (loss_fn(target1, tf.cast(outputs1, tf.float32)) +
                loss_fn(target2, tf.cast(outputs2, tf.float32)) +
                loss_fn(target3, tf.cast(outputs3, tf.float32)) +
                loss_fn(target4, tf.cast(outputs4, tf.float32)))
        if hasattr(optimizer, 'get_scaled_loss'):
            # TF <= 2.10 LossScaleOptimizer: manually scale the loss.
            # The scaling must happen inside the tape; otherwise the
            # multiplication is not recorded and the gradients are all None.
            scaled_loss = optimizer.get_scaled_loss(loss)
        else:
            # TF >= 2.11 LossScaleOptimizer: apply_gradients handles scaling internally.
            scaled_loss = loss
    grads = tape.gradient(scaled_loss, model.trainable_variables)
    if hasattr(optimizer, 'get_scaled_loss'):
        # The old version requires unscaling the gradients before the update.
        grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


for idx in range(10):
    # Use a new random initialization and empty optimizer state for each run.
    reset_experiment()

    # Training loop (total-step control replaces the original double loop).
    t_start = time.time()
    for step in range(total_steps):
        loss = train_step()
        if step % print_every == 0 or step == total_steps - 1:
            print(f"Step {step}/{total_steps}, Loss: {loss.numpy():.6f}")
    print(f"Training time: {time.time() - t_start:.1f}s")

    # Test prediction.
    (test_output1, test_output2, test_output3, test_output4), _ = model(inputs)

    print(test_output1.numpy().shape)
    print("Interaction weight matrix:")
    print(model.synweights.numpy())


    # Under mixed precision, model outputs are float16; cast back to float32
    # before saving (same dtype as v17).
    np.save(str(idx)+'_output1.npy', tf.cast(test_output1, tf.float32).numpy()[0, :, 0])
    np.save(str(idx)+'_output2.npy', tf.cast(test_output2, tf.float32).numpy()[0, :, 0])
    np.save(str(idx)+'_output3.npy', tf.cast(test_output3, tf.float32).numpy()[0, :, 0])
    np.save(str(idx)+'_output4.npy', tf.cast(test_output4, tf.float32).numpy()[0, :, 0])

    np.save(str(idx)+'_observation1.npy', np.mean(ca_1_exp, axis=1))
    np.save(str(idx)+'_observation2.npy', np.mean(ca_2_exp, axis=1))
    np.save(str(idx)+'_observation3.npy', np.mean(ca_3_exp, axis=1))
    np.save(str(idx)+'_observation4.npy', np.mean(ca_4_exp, axis=1))

    np.save(str(idx)+'_weight.npy', model.synweights.numpy())
