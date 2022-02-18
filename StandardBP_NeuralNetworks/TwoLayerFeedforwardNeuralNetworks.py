import Neuron
import math
import pandas as pd


# 两层前馈神经网络
class TwoLayerFeedforwardNeuralNetworks:
    def __init__(self, number_of_inputs, number_of_outputs, number_of_hidden_neurons):
        # 训练集，初始化为空列表
        self.train_set = []
        # 初始化输出神经元的个数
        self.number_of_outputs = number_of_outputs
        # 初始化隐层神经元个数
        self.number_of_hidden_neurons = number_of_hidden_neurons
        # 初始化输入的个数
        self.number_of_inputs = number_of_inputs
        # 存储神经网络的输出
        self.outputs = []
        # 存储隐层神经元的输出
        self.hidden_outputs = []
        # 建立输出神经元
        self.output_neurons = []
        for i in range(self.number_of_outputs):
            self.output_neurons.append(Neuron.Neuron(self.number_of_hidden_neurons))
        # 建立隐层神经元
        self.hidden_neurons = []
        for i in range(self.number_of_hidden_neurons):
            self.hidden_neurons.append(Neuron.Neuron(self.number_of_inputs))
        # 得到隐层神经元的连接权重
        self.hidden_connection_weights = []
        for neuron in self.hidden_neurons:
            self.hidden_connection_weights.append(neuron.get_connection_weights())
        # 得到输出神经元的连接权重
        self.output_connection_weights = []
        for neuron in self.output_neurons:
            self.output_connection_weights.append(neuron.get_connection_weights())

    # 用神经网络计算输出
    def cal_outputs(self, inputs):
        self.hidden_outputs = []
        self.outputs = []
        for neuron in self.hidden_neurons:
            self.hidden_outputs.append(neuron.get_output(inputs))
        for neuron in self.output_neurons:
            self.outputs.append(neuron.get_output(self.hidden_outputs))
        return self.outputs

    # 计算输出层神经元的梯度项g
    def calculate_g(self, results):
        g = []
        # print(len(self.outputs))
        for i in range(len(results)):
            g.append(self.outputs[i] * (1 - self.outputs[i]) * (results[i] - self.outputs[i]))
        return g

    # 计算隐层神经元的梯度项e
    def calculate_e(self, g):
        e = []
        for h in range(self.number_of_hidden_neurons):
            tmp = 0.
            for j in range(self.number_of_outputs):
                tmp = tmp + self.output_connection_weights[j][h] * g[j]
            e.append(self.hidden_outputs[h] * (1 - self.hidden_outputs[h]) * tmp)
        return e

    def update_output_connection_weights(self, g, learning_rate):
        for i in range(self.number_of_outputs):
            for j in range(self.number_of_hidden_neurons):
                delta = learning_rate * g[i] * self.hidden_outputs[j]
                self.output_connection_weights[i][j] = self.output_connection_weights[i][j] + delta

    def update_hidden_connection_weights(self, e, learning_rate, inputs):
        for h in range(self.number_of_hidden_neurons):
            for i in range(len(inputs)):
                delta = learning_rate * e[h] * inputs[i]
                self.hidden_connection_weights[h][i] = self.hidden_connection_weights[h][i] + delta

    def update_output_threshold(self, g, learning_rate):
        for j in range(self.number_of_outputs):
            delta = -learning_rate * g[j]
            for neuron in self.output_neurons:
                neuron.set_threshold(neuron.get_threshold() + delta)

    def update_hidden_threshold(self, e, learning_rate):
        for h in range(self.number_of_hidden_neurons):
            delta = -learning_rate * e[h]
            for neuron in self.output_neurons:
                neuron.set_threshold(neuron.get_threshold() + delta)

    def train(self, train_set, learning_rate, min_MSE):
        rounds = 1
        while True:
            MSE = 0.
            for sample in train_set:
                inputs = sample[0]
                results = sample[1]
                # 计算神经网络输出
                self.cal_outputs(inputs)
                # 计算输出神经元的梯度项
                g = self.calculate_g(results)
                # 计算隐层神经元的梯度项
                e = self.calculate_e(g)
                # 以下四个函数分别更新隐层、输出层的连接权重和阈值
                self.update_hidden_connection_weights(e, learning_rate, inputs)
                self.update_hidden_threshold(e, learning_rate)
                self.update_output_connection_weights(g, learning_rate)
                self.update_output_threshold(g, learning_rate)
                # 计算训练集的均方误差
                for i in range(self.number_of_outputs):
                    MSE = MSE + math.pow(self.outputs[i] - results[i], 2) / 2
            MSE = MSE / len(train_set)
            print('第', rounds, '轮，整体均方误差为：', MSE)
            rounds = rounds + 1
            # 如果整体均方误差小于设定的最小值，则退出训练
            if min_MSE > MSE:
                break

    def test(self, test_set):
        number_of_right_predictions = 0
        for sample in test_set:
            inputs = sample[0]
            results = sample[1]
            self.cal_outputs(inputs)
            predictions = []
            for out in self.outputs:
                if out < 0.5:
                    predictions.append(0)
                else:
                    predictions.append(1)
            print('预测为：', predictions)
            print('真实为：', results)
            if predictions == results:
                print('    预测正确')
                number_of_right_predictions = number_of_right_predictions + 1
            else:
                print('  预测失败')
        accuracy = number_of_right_predictions / len(test_set)
        print('预测正确个数为：', number_of_right_predictions)
        print('预测准确率为：', accuracy * 100, '%')


def deal_with_results(ori_results):
    dct = {}
    count = 0
    for i in range(len(ori_results)):
        if not ori_results[i] in dct:
            dct[ori_results[i]] = count
            count = count + 1
    for i in range(len(ori_results)):
        tmp = [0.] * count
        tmp[dct[ori_results[i]]] = 1.
        ori_results[i] = tmp


#
# 鸢尾花数据集
#
original_form = pd.read_csv('/Users/mac/Downloads/iris.csv')
original_inputs = original_form.iloc[:, 0:4].values.tolist()
original_results = original_form.iloc[:, 4].values.tolist()
deal_with_results(original_results)
print(original_results)
data_set = []
for t in range(len(original_results)):
    data_set.append([original_inputs[t], original_results[t]])
print(data_set)
train_set = data_set[10:50] + data_set[60:100] + data_set[110:150]
print(train_set)
iris = TwoLayerFeedforwardNeuralNetworks(4, 3, 4)
iris.train(train_set, 0.09, 0.02)
test_set = data_set[0:10] + data_set[50:60] + data_set[100:110]
iris.test(test_set)

#
# uci乳腺癌数据集
#
original_form = pd.read_csv('/Users/mac/Downloads/breast-cancer-wisconsin.csv')
original_inputs = original_form.iloc[:, 1:10].values.tolist()
original_results = original_form.iloc[:, 10].values.tolist()
deal_with_results(original_results)
print(original_inputs)
data_set = []
for t in range(len(original_results)):
    data_set.append([original_inputs[t], original_results[t]])
print(data_set)
train_set = data_set[0:500]
print(train_set)
breast_cancer = TwoLayerFeedforwardNeuralNetworks(9, 2, 10)
breast_cancer.train(train_set, 0.02, 0.045)
test_set = data_set[500:699]
breast_cancer.test(test_set)
