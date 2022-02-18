import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd


# 对数几率函数"Sigmoid函数"
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 计算β的一阶导
def d_beta(beta, X, y):
    ret = np.zeros((X.shape[0], 1))
    for i in range(X.shape[1]):
        x = np.array([X[:, i]]).T
        beta_T_dot_x = np.dot(beta.T[0], x)
        p = sigmoid(beta_T_dot_x[0])
        ret = ret - np.dot(x, y[i] - p)
    return ret


# 计算β的二阶导
def d2_beta(beta, X, y):
    ret = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[1]):
        x = np.array([X[:, i]]).T
        beta_T_dot_x = np.dot(beta.T[0], x)
        p = sigmoid(beta_T_dot_x[0])
        ret = ret + np.dot(np.dot(x, x.T), p * (1 - p))
    return ret


# 用牛顿法迭代一轮，得到新的β
def a_newton_iteration(old_beta, X, y):
    new_beta = old_beta - np.dot(np.linalg.inv(d2_beta(old_beta, X, y)), d_beta(old_beta, X, y))
    return new_beta


# 计算l(β)
def l_beta(beta, X, y):
    ret = 0.0
    for i in range(X.shape[1]):
        x = np.array([X[:, i]]).T
        beta_T_dot_x = np.dot(beta.T[0], x)
        ret = ret - y[i] * beta_T_dot_x[0] + np.log(1 + np.exp(beta_T_dot_x[0]))
    return ret


# 用牛顿法迭代多轮，直到误差小于10的-8次方，返回此时的β参数
def a_logistic_regression(X, y):
    beta = np.array([[0.] * X.shape[0]]).T
    old_l_beta = l_beta(beta, X, y)
    while (1):
        beta = a_newton_iteration(beta, X, y)
        new_l_beta = l_beta(beta, X, y)
        if np.abs(new_l_beta - old_l_beta) < 1e-8:
            break
        old_l_beta = new_l_beta
    return beta


def return_attributes(original_list, dict):
    return_list = []
    for item in original_list:
        return_list.append(dict[item])
    return return_list


# 用分类器计算得到西瓜测试集的分类结果
def calculate_test_results_for_melon(beta, X):
    results = []
    for i in range(X.shape[1]):
        x = X[:, [i]]
        beta_T_dot_x = np.dot(beta.T[0], x)
        p = sigmoid(beta_T_dot_x[0])
        if p > 0.5:
            results.append(1)
        else:
            results.append(0)
    return results


# 用分类器计算得到鸢尾花测试集的分类结果
def calculate_test_results_for_iris(beta1, beta2, beta3, X):
    results = []
    dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    for i in range(X.shape[1]):
        x = X[:, [i]]
        # result列表存放三个分类器的预测结果
        result = []
        result.append(sigmoid(np.dot(beta1.T[0], x)[0]))
        result.append(sigmoid(np.dot(beta2.T[0], x)[0]))
        result.append(sigmoid(np.dot(beta3.T[0], x)[0]))
        if result[0] > 0.5:
            comp = 0
        elif result[1] > 0.5:
            comp = 1
        else:
            comp = 2
        results.append(dict[comp])
    return results


# 计算分类器分类结果的正确率
def calculate_accuracy(calculated_results, original_results):
    sum = 0.
    for i in range(len(original_results)):
        if original_results[i] == calculated_results[i]:
            sum = sum + 1
    return sum / len(original_results)


# 接下来的程序将西瓜数据集划分为训练集与测试集，然后用训练集训练，并用测试集统计正确率
input_file = '/Users/mac/Downloads/melon.csv'
original_form = pd.read_csv(input_file, sep=',', encoding='utf-8')
# 这些字典用于将各个属性转化为数字
list_of_dicts = [{'浅白':0., '青绿':0.2, '乌黑':0.4, 'attribute_name':'色泽'},
                 {'蜷缩':0., '稍蜷':0.2, '硬挺':0.4, 'attribute_name':'根蒂'},
                 {'清脆':0., '浊响':0.2, '沉闷':0.4, 'attribute_name':'敲声'},
                 {'清晰':0., '稍糊':0.2, '模糊':0.4, 'attribute_name':'纹理'},
                 {'平坦':0., '稍凹':0.2, '凹陷':0.4, 'attribute_name':'脐部'},
                 {'硬滑':0., '软粘':0.4, 'attribute_name':'触感'}]
X = []
for dict in list_of_dicts:
    X.append(return_attributes(list(original_form[dict['attribute_name']]), dict))
X.append(list(original_form[u'密度']))
X.append(list(original_form[u'含糖率']))
X.append([1.] * 17)
# 此时X矩阵与y矩阵中存着所有数据，但并未划分 训练集 与 测试集，接下来将会划分
X = np.array(X)
y = np.array([original_form[u'序关系']])
# 按照训练集占总数的2/3～4/5的原则进行划分，取6组正例与6组反例为训练集
X_for_train = X[:, [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]]
y_for_train = y[:, [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]][0]
# 取剩下的数据为测试集
X_for_test = X[:, [6, 7, 8, 9, 10]]
print('X_for_test:', X_for_test)
y_for_test = y[:, [6, 7, 8, 9, 10]][0]
# 计算得到β向量
beta = a_logistic_regression(X_for_train, y_for_train)
print('西瓜数据集 训练集 解出的Beta列向量:\n', beta)
print('西瓜数据集 测试集 真实分类结果为：\n', y_for_test)
# 用得到的分类器算出分类结果
calculated_results = calculate_test_results_for_melon(beta, X_for_test)
print('西瓜数据集 测试集 算出分类结果为：\n', calculated_results)
print('西瓜数据集 测试集 分类的正确率为：\n', calculate_accuracy(calculated_results, y_for_test) * 100, '%\n')

# 接下来的程序将鸢尾花数据集划分为训练集与测试集，然后用训练集训练，并用测试集统计正确率
input_file = '/Users/mac/Downloads/iris.csv'
original_form = pd.read_csv(input_file, sep=',', encoding='utf-8')
X = []
X.append(list(original_form['sepal_length']))
X.append(list(original_form['sepal_width']))
X.append(list(original_form['petal_length']))
X.append(list(original_form['petal_width']))
X.append([1.] * 150)
X = np.array(X)
y = [1.] * 40 + [0.] * 80
# 采用OvR模式进行多分类
# 分类器1的训练集，用于识别种类是否为setosa，每种类别的数据都取4/5用于训练
X_for_train_1 = X[:, list(range(0, 40)) + list(range(50, 90)) + list(range(100, 140))]
beta_1 = a_logistic_regression(X_for_train_1, y)
# 分类器2的训练集，用于识别种类是否为versicolor，每种类别的数据都取4/5用于训练
X_for_train_2 = X[:, list(range(50, 90)) + list(range(0, 40)) + list(range(100, 140))]
beta_2 = a_logistic_regression(X_for_train_2, y)
# 分类器3的训练集，用于识别种类是否为virginica，每种类别的数据都取4/5用于训练
X_for_train_3 = X[:, list(range(100, 140)) + list(range(0, 40)) + list(range(50, 90))]
beta_3 = a_logistic_regression(X_for_train_3, y)
print('鸢尾花数据集 训练集 分类器1～3的beta分别为：\n', beta_1, '\n\n', beta_2, '\n\n', beta_3, '\n')
X_for_test_iris = X[:, list(range(40, 50)) + list(range(90, 100)) + list(range(140, 150))]
# 用三个分类器计算分类结果
calculated_results = calculate_test_results_for_iris(beta_1, beta_2, beta_3, X_for_test_iris)
original_results = ['setosa'] * 10 + ['versicolor'] * 10 + ['virginica'] * 10
print('鸢尾花数据集 测试集 原始分类结果为：\n', original_results)
print('鸢尾花数据集 测试集 预测分类结果为：\n', calculated_results)
print('鸢尾花数据集 测试集 分类的正确率为：\n', calculate_accuracy(calculated_results, original_results) * 100, '%')
