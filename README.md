# BPNN
反向传播神经网络
## 1. 引例
### 1.1 数据说明
train.csv中包括约1000条的医疗账单数据，其中
| age | sex | bmi | children | smoker | region | charges |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 45 | female | 25.175 | 2 | no | northeast | 9095.06825 |
| 19 | male | 31.92 | 0 | yes | northwest | 33750.2918 |  
###
**age**：年龄  
**sex**：性别，包括male和female  
**bmi**：身体质量指数  
**children**：孩子的数量  
**somker**：是否吸烟，yes表示吸烟，no表示不吸烟  
**region**：所属地区，包括northeast，northwest，southeast和southwest四个地区  
**charges**： 医疗费用  
基于上述未加粗属性（特征）构建模型预测charges属性  

### 1.2 模型要求
使用pandas读取数据train.csv，自行决定数据的预处理方式
使用numpy构建全连接神经网络，自行决定模型结构
## 2. 分析
### 2.1 数据预处理
将特征中的文本信息转化为数字编码  
对数据进行归一化或标准化，防止神经网络偏置
### 2.2 网络结构
考虑一个简单的网络，拥有一层隐藏层，隐藏层节点数为7，输入节点6个，输出节点1个。
### 2.3 反向传播算法
反向传播算法(Back propagation)是“误差反向传播”的简称，是适合于多层神经元网络的一种学习算法，它建立在梯度下降法的基础上。梯度下降法是训练神经网络的常用方法，许多的训练方法都是基于梯度下降法改良出来的，因此了解梯度下降法很重要。梯度下降法通过计算损失函数的梯度，并将这个梯度反馈给最优化函数来更新权重以最小化损失函数。
BP算法的学习过程由正向传播过程和反向传播过程组成。  
它的基本思想为：
(1)先计算每一层的状态和激活值，直到最后一层（即信号是前向传播的）；
(2)计算每一层的误差，误差的计算过程是从最后一层向前推进的（即误差是反向传播的）；
(3)计算每个神经元连接权重的梯度；
(4)根据梯度下降法则更新参数（目标是误差变小）。
迭代以上步骤，直到满足停止准则（比如相邻两次迭代的误差的差别很小）。
## 3. 实现细节  
### 3.1 符号系统
$x$：输入向量，表示输入数据，每个元素对应一个输入特征  
$\hat y$：输出向量，表示网络的预测输出
$y$：目标向量，表示实际的输出值
$W$：权重矩阵，表示连接层与层之间的权重  
$𝑏$：偏置向量，表示每一层神经元的偏置项  
$\sigma$：激活函数，用于引入非线性，如sigmoid函数、ReLU函数等  
$\eta$：学习率，用于控制权重更新步长的超参数  
$z$：隐藏层输出向量，表示隐藏层的输出  
$\delta$：误差项，表示误差的梯度，用于反向传播过程中  
$𝑎$：激活值，表示经过激活函数处理后的输出  
$W^{(l)}$ ：第 $𝑙$层的权重矩阵  
$b^{(l)}$ ：第 $𝑙$层的偏置向量  
$\delta^{(l)}$：第 $𝑙$ 层的误差项  
$𝐸$：损失函数 $𝐸=\frac{1}{2}(\hat 𝑦−y)^2$  
### 3.2 初始化神经网络
```python
input_size = np.size(x, 1)  # 输入特征数:6
hidden_size = 7  # 隐藏层大小
output_size = 1  # 输出大小（医疗费用）
```
```python
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
```
### 3.3 前向传播
```python
    def forward(self, x):
        # 前向传播
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output
```
隐藏层输出： $𝑧=\delta(𝑊^{(1)}𝑥+𝑏^{(1)})$  
输出层输出： $𝑦=𝑊^{(2)}𝑧+𝑏^{(2)}$  
### 3.4 反向传播
```python
    def backward(self, x, y, output, learning_rate, i):
        # 计算输出层的误差(偏置梯度)
        output_error = output - y  # loss = 0.5 * mse_loss(self.output[i, :], y[i, :].T)
        # 计算输出层权重的梯度
        output_delta = np.atleast_2d(output_error * self.hidden_output[i, :]).T  # 7*1
        # 计算隐藏层的误差(偏置梯度)
        hidden_error = np.dot(self.weights_hidden_output, np.atleast_2d(output_error)).T * sigmoid_derivative(
            self.hidden_input[i, :])  # 1*7
        # 计算隐藏层权重的梯度
        hidden_delta = hidden_error * np.atleast_2d(x).T  # 6*7 hidden_delta = np.dot(np.atleast_2d(x[i, :]).T, hidden_error)

        # 更新权重和偏置
        self.weights_hidden_output -= learning_rate * output_delta  # 7*1
        self.bias_hidden_output -= learning_rate * output_error  # 1*1
        self.weights_input_hidden -= learning_rate * hidden_delta  # 6*7
        self.bias_input_hidden -= learning_rate * hidden_error  # 1*7
```
误差分析：  
输出层误差： 即损失函数的梯度 $\nabla E=𝛿^{(2)}=\hat 𝑦−y$  
隐藏层误差： $𝛿^{(1)}=(𝑊^{(2)})^⊤𝛿^{(2)}⊙𝜎^′(𝑥)$  
权重更新：  
输出层权重更新： $𝑊^{(2)}\leftarrow 𝑊^{(2)}−𝜂𝛿^{(2)}𝑧^⊤$  
输出层偏置更新： $b^{(2)}\leftarrow b^{(2)}-𝜂𝛿^{(2)}$  
隐藏层权重更新： $𝑊^{(1)}\leftarrow 𝑊^{(1)}-𝜂𝛿^{(1)}𝑥^⊤$  
隐藏层偏置更新： $b^{(1)}\leftarrow b^{(1)}-𝜂𝛿^{(1)}$  
