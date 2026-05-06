import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

###################
# RNN_interaction_modeling_GPU_limit_edge.py
# The model consists of four independent LSTM subnetworks, each corresponding to one neuronal type. 
# Interactions between subnetworks were implemented via directed synaptic connections.
#  Based on prior knowledge of connectivity patterns between excitatory and inhibitory neurons, we initialized all connections. 
# The only constraint was that VIP neurons could only receive unidirectional connections from excitatory neurons, without exerting a direct effect on excitatory neurons. 
# During training, inputs were delivered to all four subnetworks in the time interval [ -3, -2] seconds relative to onset, and each subnetwork generated predicted activity.
#  By comparing these predicted values with the actual recorded neural activity data, the mean squared error (MSE) is calculated as the loss function, 
# and the network parameters (including connection weights) are updated through the backpropagation algorithm. 
# Ultimately, the optimized interaction RNN model is not only capable of predicting the average activity signals of the four types of neurons,
#  but also predicting the connection weights within the network which reflect the directed interaction relationships between the neurons.
# 为了避免每次初始化连接权重带来的随机性，我们拟合了10次，并去平均值代表连接权重。


# 检查GPU可用性
print("GPU 可用:", tf.config.list_physical_devices('GPU'))

# 配置GPU内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    # 或者也可以设置GPU显存为固定使用量(例如：4G)
    #tf.config.experimental.set_virtual_device_configuration(gpu0,
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    tf.config.set_visible_devices([gpu0],"GPU")
# 启用混合精度训练
#from tensorflow.keras import  mixed_precision
#policy = tf.keras.mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)
#print('Compute dtype:', policy.compute_dtype)
#print('Variable dtype:', policy.variable_dtype)
class InteractiveRNN(tf.keras.Model):
    def __init__(self, units, input_dim):
        super(InteractiveRNN, self).__init__()
        self.units = units
        self.input_dim = input_dim

        # 定义四个RNN核心（使用return_sequences=True）
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

        # 添加Dense层以压缩输出到1维
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(1)
        self.dense3 = tf.keras.layers.Dense(1)
        self.dense4 = tf.keras.layers.Dense(1)

        # 定义可训练标量权重矩阵（4x4）
        # 初始化为0.5，对角线为0（自身不作用于自身）
        initial_weights = np.full((4, 4), -0.5)
        initial_weights[:, 0] = 0.5
        #initial_weights[2, 1] = 0
        #initial_weights[3, 1] = 0
        initial_weights[0, 3] = 0
        np.fill_diagonal(initial_weights, 0.0)

        # 转换为Tensor
        initial_weights = tf.constant(initial_weights, dtype=tf.float32)

        # 定义可训练权重矩阵
        self.synweights = tf.Variable(initial_weights, trainable=True, name='weights')

    def call(self, inputs, initial_states=None):
        # 解包输入
        input1, input2, input3, input4 = inputs

        # 初始化状态时自动适配batch大小
        batch_size = tf.shape(input1)[0]  # 动态获取batch大小

        # 初始化四个RNN的状态
        if initial_states is None:
            state1 = [tf.zeros((batch_size, self.units)),
                      tf.zeros((batch_size, self.units))]
            state2 = [tf.zeros((batch_size, self.units)),
                      tf.zeros((batch_size, self.units))]
            state3 = [tf.zeros((batch_size, self.units)),
                      tf.zeros((batch_size, self.units))]
            state4 = [tf.zeros((batch_size, self.units)),
                      tf.zeros((batch_size, self.units))]
        else:
            state1, state2, state3, state4 = initial_states

        # 初始化第一次pre_output
        prev_output1 = tf.zeros((tf.shape(input1)[0], 1))
        prev_output2 = tf.zeros((tf.shape(input1)[0], 1))
        prev_output3 = tf.zeros((tf.shape(input1)[0], 1))
        prev_output4 = tf.zeros((tf.shape(input1)[0], 1))

        # 存储所有时间步的输出
        all_outputs1, all_outputs2, all_outputs3, all_outputs4 = [], [], [], []

        # 按时间步处理
        for t in range(input1.shape[1]):
            # 保存旧的状态
            prev_state1 = state1
            prev_state2 = state2
            prev_state3 = state3
            prev_state4 = state4

            # 获取当前时间步输入
            x1_t = input1[:, t, :]
            x2_t = input2[:, t, :]
            x3_t = input3[:, t, :]
            x4_t = input4[:, t, :]



            # RNN1处理（接收其他RNN的加权状态）
            x1_t = x1_t + \
                        self.synweights[0, 1] * prev_output2 + \
                        self.synweights[0, 2] * prev_output3
            output1, new_state1_h, new_state1_c = self.rnn1(
                tf.expand_dims(x1_t, 1),
                initial_state=prev_state1
            )
            state1 = [new_state1_h, new_state1_c]
            # RNN2处理（接收其他RNN的加权状态）
            x2_t = x2_t + \
                           self.synweights[1, 0] * prev_output1 + \
                           self.synweights[1, 2] * prev_output3 + \
                           self.synweights[1, 3] * prev_output4
            output2, new_state2_h, new_state2_c  = self.rnn2(
                tf.expand_dims(x2_t, 1),
                initial_state=prev_state2
            )
            state2 = [new_state2_h, new_state2_c]
            # RNN3处理（接收其他RNN的加权状态）
            x3_t = x3_t + \
                           self.synweights[2, 0] * prev_output1 + \
                           self.synweights[2, 1] * prev_output2 + \
                           self.synweights[2, 3] * prev_output4
            output3, new_state3_h, new_state3_c = self.rnn3(
                tf.expand_dims(x3_t, 1),
                initial_state=prev_state3
            )
            state3 = [new_state3_h, new_state3_c]
            # RNN4处理（接收其他RNN的加权状态）
            x4_t = x4_t + \
                           self.synweights[3, 0] * prev_output1 + \
                           self.synweights[3, 1] * prev_output2 + \
                           self.synweights[3, 2] * prev_output3

            output4, new_state4_h, new_state4_c = self.rnn4(
                tf.expand_dims(x4_t, 1),
                initial_state=prev_state4
            )
            state4 = [new_state4_h, new_state4_c]
            # 使用Dense层压缩输出到1维
            output1 = self.dense1(output1)
            output2 = self.dense2(output2)
            output3 = self.dense3(output3)
            output4 = self.dense4(output4)

            prev_output1, prev_output2, prev_output3, prev_output4 = output1[:, 0, :], output2[:, 0, :], output3[:, 0, :], output4[:, 0, :]
            # 收集输出（移除时间步维度）
            all_outputs1.append(output1[:, 0, :])
            all_outputs2.append(output2[:, 0, :])
            all_outputs3.append(output3[:, 0, :])
            all_outputs4.append(output4[:, 0, :])

        # 合并时间步输出
        outputs1 = tf.stack(all_outputs1, axis=1)  # shape: (batch, time, 1)
        outputs2 = tf.stack(all_outputs2, axis=1)
        outputs3 = tf.stack(all_outputs3, axis=1)
        outputs4 = tf.stack(all_outputs4, axis=1)

        # 返回格式: (outputs1, outputs2, outputs3, outputs4), (state1, state2, state3, state4)
        return (outputs1, outputs2, outputs3, outputs4), (state1, state2, state3, state4)

# 参数设置
input_dim = 1
units = 50
batch_size = 32
time_steps = 200  # 设置为199
sample_num = 540
total_epoch = 5


for idx in range(10):

    # 读取数据
    data = np.load("S2I_new_example_zscore.npy")



    ca_1_exp = data[0,:,:]
    ca_2_exp = data[1,:,:]
    ca_3_exp = data[2,:,:]
    ca_4_exp = data[3,:,:]
    target1 = ca_1_exp.reshape(1, time_steps, sample_num )
    target2 = ca_2_exp.reshape(1, time_steps, sample_num )
    target3 = ca_3_exp.reshape(1, time_steps, sample_num )
    target4 = ca_4_exp.reshape(1, time_steps, sample_num )


    # 输入数据
    t = np.linspace(0, 10, time_steps)

    input1 = np.zeros((1, time_steps, 1))
    input1[0, 20:30, 0] = 1

    input2 = np.zeros((1, time_steps, 1))
    input2[0, 20:30, 0] = 1

    input3 = np.zeros((1, time_steps, 1))
    input3[0, 20:30, 0] = 1

    input4 = np.zeros((1, time_steps, 1))
    input4[0, 20:30, 0] = 1






    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    model = InteractiveRNN(units, input_dim)
    optimizer = tf.keras.optimizers.Adam(0.005)
    loss_fn = tf.keras.losses.MeanSquaredError()
    #loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # 将输入数据转换为GPU张量
    input1 = tf.convert_to_tensor(input1, dtype=tf.float32)
    input2 = tf.convert_to_tensor(input2, dtype=tf.float32)
    input3 = tf.convert_to_tensor(input3, dtype=tf.float32)
    input4 = tf.convert_to_tensor(input4, dtype=tf.float32)
    # 将输入数据转换为GPU张量
    target1 = tf.convert_to_tensor(target1, dtype=tf.float32)
    target2 = tf.convert_to_tensor(target2, dtype=tf.float32)
    target3 = tf.convert_to_tensor(target3, dtype=tf.float32)
    target4 = tf.convert_to_tensor(target4, dtype=tf.float32)

    # 正确解包返回结果
    (outputs1, outputs2, outputs3, outputs4), (state1, state2, state3, state4) = model((input1, input2, input3, input4))

    print("输出1形状:", outputs1.shape)  # 应为 (1, 199, 1)
    print("输出2形状:", outputs2.shape)
    print("输出3形状:", outputs3.shape)
    print("输出4形状:", outputs4.shape)



    # 训练循环
    for epoch in range(total_epoch):
        epoch_loss = 0.0
        for i in range(sample_num):
            with tf.GradientTape() as tape:
                target1_temp = target1[0, :, 0]
                target2_temp = target2[0, :, 0]
                target3_temp = target3[0, :, 0]
                target4_temp = target4[0, :, 0]
                # 前向传播
                (outputs1, outputs2, outputs3, outputs4), _ = model(
                    (input1, input2, input3, input4))

                # 计算损失（自动广播到batch）
                loss = (loss_fn(target1, outputs1) +
                        loss_fn(target2, outputs2) +
                        loss_fn(target3, outputs3) +
                        loss_fn(target4, outputs4))

            # 梯度计算与更新
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            # 打印epoch级损失
            print(f"Epoch {epoch}, Batch {i}, Avg Loss: {loss}")



    # 测试预测
    (test_output1, test_output2, test_output3, test_output4), _ = model((input1, input2, input3, input4))

    print(test_output1.numpy().shape)
    print("交互权重矩阵:")
    print(model.synweights.numpy())

    np.save(str(idx)+'_output1.npy', test_output1.numpy()[0, :, 0])
    np.save(str(idx)+'_output2.npy', test_output2.numpy()[0, :, 0])
    np.save(str(idx)+'_output3.npy', test_output3.numpy()[0, :, 0])
    np.save(str(idx)+'_output4.npy', test_output4.numpy()[0, :, 0])

    np.save(str(idx)+'_obervation1.npy', np.mean(ca_1_exp, axis=1))
    np.save(str(idx)+'_obervation2.npy', np.mean(ca_2_exp, axis=1))
    np.save(str(idx)+'_obervation3.npy', np.mean(ca_3_exp, axis=1))
    np.save(str(idx)+'_obervation4.npy', np.mean(ca_4_exp, axis=1))

    np.save(str(idx)+'_weight.npy', model.synweights.numpy())