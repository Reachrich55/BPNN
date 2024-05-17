import numpy as np
import pandas as pd

# Adam 优化器
class Adam:
    def __init__(self, learning_rate):
        # 初始化Adam优化器
        self.learning_rate = learning_rate
        # beta1: 第一动量的指数衰减率，默认为0.9
        self.beta1 = 0.9
        # beta2: 第二动量的指数衰减率，默认为0.999
        self.beta2 = 0.999

        # epsilon: 用于防止除零错误的小数，默认为1e-10
        self.epsilon = 1e-10
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # He初始化权重
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)

        # 初始化偏置为零
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.bias_hidden_output = np.zeros((1, output_size))

        # 初始化优化器
        self.optimizer = Adam(learning_rate=0.01)

    # 前向传播
    def forward(self, X):

        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = relu(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output


    # 反向传播
    def backward(self, X, Y):
        # self.backward(X[i, :], y[i, :], self.output[i, :], i)
        # 计算输出层的误差
        error = self.output - Y

        # 计算输出层权重的梯度
        output_weights_delta = np.dot(self.hidden_output.T, error)
        output_bias_delta = np.sum(error, axis=0, keepdims=True)

        # 计算隐藏层的误差
        hidden_error = np.dot(error, self.weights_hidden_output.T) * relu_derivative(self.hidden_input)

        # 计算隐藏层权重的梯度
        hidden_weights_delta = np.dot(X.T, hidden_error)
        hidden_bias_delta = np.sum(hidden_error, axis=0, keepdims=True)

        # 更新权重和偏置
        grads = [output_weights_delta, output_bias_delta, hidden_weights_delta, hidden_bias_delta]
        self.optimizer.update([self.weights_hidden_output, self.bias_hidden_output, self.weights_input_hidden, self.bias_input_hidden],grads)

    def train(self, X, Y, epochs):
        batch_size = 64
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                Y_batch = Y[i:i + batch_size]
                # 前向传播
                self.output = self.forward(X_batch)
                # 反向传播
                self.backward(X_batch, Y_batch)
# 定义ReLU激活函数及其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def train_neural_network(input_size, hidden_size, output_size, X_train, y_train, epochs):
    # 实例化神经网络对象
    neural_net = NeuralNetwork(input_size, hidden_size, output_size)

    # 训练神经网络
    neural_net.train(X_train, y_train, epochs)

    return neural_net

# 预处理数据
def pretreatment(data):
    # 对sex和smoker进行二元编码
    data['sex'] = (data['sex'] == 'male').astype(int)
    data['smoker'] = (data['smoker'] == 'yes').astype(int)

    # 对region进行编码
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    data['region'] = data['region'].map(region_map)

    # 对数值特征进行归一化
    numerical_features = ['age', 'children', 'bmi']
    data[numerical_features] = (data[numerical_features] - data[numerical_features].min()) / (data[numerical_features].max() - data[numerical_features].min())

    return data

# 读取数据
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')

# 预处理数据
pretreatment(train_data)
pretreatment(val_data)

# 提取特征和标签
X_train = train_data.drop(columns=['charges']).to_numpy()
y_train = train_data['charges'].to_numpy()
y_train = np.atleast_2d(y_train).T

X_val = val_data.drop(columns=['charges']).to_numpy()
y_val = val_data['charges'].to_numpy()
y_val = np.atleast_2d(y_val).T

# 定义神经网络参数
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1

# 训练神经网络
neural_net = train_neural_network(input_size, hidden_size, output_size, X_train, y_train, epochs=8000)

# 使用训练后的神经网络进行预测
y_pred = neural_net.forward(X_val)

# 计算预测值与真实值之间的损失
val_loss = np.mean(np.abs(y_pred - y_val))
print(val_loss)
