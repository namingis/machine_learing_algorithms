import math
import random


# 神经元类
class Neuron:
    def __init__(self, number_of_inputs):
        self.number_of_inputs = number_of_inputs
        # 初始化连接权重设置
        self.connection_weights = []
        for i in range(number_of_inputs):
            self.connection_weights.append(random.random())
        # 初始化神经元阈值
        self.threshold = random.random()
        self.output = 0
        print('threshold:', self.threshold)
        print('weights:', self.connection_weights)

    # 此处用sigmoid函数作为激活函数
    def activation_function(self, x):
        return 1. / (1. + math.exp(-x))

    # 得到该神经元的输出
    def get_output(self, inputs):
        ret = 0.
        for i in range(self.number_of_inputs):
            ret = ret + inputs[i] * self.connection_weights[i]
        ret = self.activation_function(ret)
        self.output = ret
        return self.output

    # 得到神经元的阈值
    def get_threshold(self):
        return self.threshold

    # 得到神经元的连接权重
    def get_connection_weights(self):
        return self.connection_weights

    # 设置神经元的阈值
    def set_threshold(self, new_threshold):
        self.threshold = new_threshold
