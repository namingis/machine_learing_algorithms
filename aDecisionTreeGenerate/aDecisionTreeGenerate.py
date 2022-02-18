import random

import numpy as np
import copy
import pandas as pd
import xlrd


def gini(D):
    dict = {}
    for item in D:
        if not dict.__contains__(item[1]):
            dict[item[1]] = 0.
        dict[item[1]] = dict[item[1]] + 1.
    g = 1.
    for val in dict.values():
        p = val / len(D)
        g = g - p * p
    return g


def gini_index(D, a):
    index = 0
    dict = {}
    for item in D:
        if not dict.__contains__(item[0][a]):
            dict[item[0][a]] = []
        dict[item[0][a]].append(item)
    for Dv in dict.values():
        index = index + len(Dv) / len(D) * gini(Dv)
    return index


# 计算训练集D的信息熵
def information_entropy(D):
    dict = {}
    for item in D:
        if not dict.__contains__(item[1]):
            dict[item[1]] = 0.
        dict[item[1]] = dict[item[1]] + 1.
    ent = 0.
    for val in dict.values():
        p = val / len(D)
        ent = ent - p * np.log2(p)
    return ent


# 计算用属性a划分训练集D的信息增益
def information_gain(D, a):
    gain = information_entropy(D)
    dict = {}
    for item in D:
        if not dict.__contains__(item[0][a]):
            dict[item[0][a]] = []
        dict[item[0][a]].append(item)
    for Dv in dict.values():
        gain = gain - len(Dv) / len(D) * information_entropy(Dv)
    return gain


# 决策树结点类
class Node:
    def __init__(self):
        # 用字典存放子结点们，要寻找某子结点，只需知道对应属性值，在字典中寻找
        self.sons = {}
        self.is_leaf = False
        self.value = ''
        # D矩阵存放传入该结点的训练集数据
        self.D = np.array([])

    def add_child(self, child, attribute_val):
        self.sons[attribute_val] = child

    # value变量在叶结点存放分类结果，其余结点存放用于划分的属性名
    def set_value(self, value):
        self.value = value

    def set_is_leaf(self, bool=True):
        self.is_leaf = bool

    def set_D(self, D):
        self.D = D


# 检查D训练集中的样本是否都是同一类型的，是则返回True
def if_D_the_same_category(D):
    ret = True
    for i in range(1, len(D)):
        if D[i][1] != D[i - 1][1]:
            ret = False
            break
    return ret


# 判断D训练集中的样本在属性集上的取值是否完全一样，是则返回True
def if_D_the_same_A_value(D):
    ret = True
    for i in range(1, len(D)):
        if D[i][0] != D[i - 1][0]:
            ret = False
            break
    return ret


# 找到D训练集中最多的类别并返回类别名
def find_the_most_category_in_D(D):
    dict = {}
    for item in D:
        if not dict.__contains__(item[1]):
            dict[item[1]] = 0.
        dict[item[1]] = dict[item[1]] + 1.
    maxi = 0.
    ret = ''
    for k, v in dict.items():
        if v > maxi:
            maxi = v
            ret = k
    return ret


# 找到最优划分属性
def find_the_best_attribute(D, A):
    max_gain = 0.
    for k in A.keys():
        gain_of_k = information_gain(D, k)
        if gain_of_k > max_gain:
            max_gain = gain_of_k
            ret = k
    return ret


# 找到训练集D中的满足在a属性的取值为val的所有样本，并组成子集Dv返回
def generate_Dv(D, a, val):
    Dv = []
    for item in D:
        if item[0][a] == val:
            Dv.append(item)
    return Dv


# 生成决策树
def aTreeGenerate(D, A):
    deal_with_continuous_attributes(D, A)
    node = Node()
    node.set_D(D)
    if if_D_the_same_category(D):
        node.set_value(D[0][1])
        node.set_is_leaf()
        return node
    if len(A) == 0 or if_D_the_same_A_value(D):
        node.set_is_leaf()
        the_most_category_in_D = find_the_most_category_in_D(D)
        node.set_value(the_most_category_in_D)
        return node
    the_best_attribute = find_the_best_attribute(D, A)
    for attribute_val in A[the_best_attribute]:
        Dv = generate_Dv(D, the_best_attribute, attribute_val)
        sub_A = copy.deepcopy(A)
        sub_A.pop(the_best_attribute)
        if len(Dv) == 0:
            a_sub_node = Node()
            a_sub_node.set_is_leaf()
            the_most_category_in_D = find_the_most_category_in_D(D)
            a_sub_node.set_value(the_most_category_in_D)
            node.add_child(a_sub_node, attribute_val)
        else:
            node.add_child(aTreeGenerate(Dv, sub_A), attribute_val)
    node.set_value(the_best_attribute)
    return node


# 读取鸢尾花数据集文件
def read_file(file_path):
    original_form = pd.read_excel(file_path)
    attributes_vals_of_samples = original_form.iloc[:, 0:4].to_dict('records')
    D = []
    print(attributes_vals_of_samples)
    categories = list(original_form.iloc[:, -1])
    print(categories)
    for i in range(len(attributes_vals_of_samples)):
        D.append([attributes_vals_of_samples[i], categories[i]])
    A = {}
    attributes_names = list(original_form)
    attributes_names.remove('species')
    for name in attributes_names:
        A[name] = np.unique(list(original_form[name]))
    print(A)
    return [D, A]


# 用决策树决定一个样本的类别
def categorize_a_sample(root, attributes_of_a_sample):
    while not root.is_leaf:
        attribute_name = root.value
        root = root.sons[attributes_of_a_sample[attribute_name]]
    return root.value


# 计算决策树在测试集上的精度（正确率）
def calculate_accuracy(D, decision_tree_root):
    categorize_results = []
    true_categories = []
    accuracy = 0.
    for item in D:
        categorize_results.append(categorize_a_sample(decision_tree_root, item[0]))
        true_categories.append(item[1])
    for i in range(len(true_categories)):
        if true_categories[i] == categorize_results[i]:
            accuracy = accuracy + 1.
    accuracy = accuracy / len(true_categories)
    print('验证集的分类器分类结果：', categorize_results)
    print('验证集上样本的真实分类：', true_categories)
    print('分类器在验证集上的正确率：', accuracy * 100, '%')
    return accuracy


# 请忽略，验证用的函数，用于展示树的结构
def display_tree(root):
    nodes = []
    while len(nodes) != 0:
        node = nodes.pop()
        node_sons = node.sons
        for k, v in node_sons.items():
            print(node.value, ':', k, '\t', v.value)
            nodes.append(v)


# 后剪枝的函数，需传入已有决策树的根
def post_pruning(root, D_for_test):
    nodes = [root]
    i = 0
    while i < len(nodes):
        node = nodes[i]
        for k, v in node.sons.items():
            nodes.append(v)
        i = i + 1
    i = len(nodes) - 1
    print('依次考察节点是否需要剪枝：')
    print()
    while i >= 0:
        if not nodes[i].is_leaf:
            print('考察节点所判断的属性：', nodes[i].value)
            original_value = nodes[i].value
            original_accuracy = calculate_accuracy(D_for_test, root)
            nodes[i].set_is_leaf()
            nodes[i].set_value(find_the_most_category_in_D(nodes[i].D))
            new_accuracy = calculate_accuracy(D_for_test, root)
            if new_accuracy > original_accuracy:
                print('    剪枝前的精度', original_accuracy * 100, '%', '小于', '剪枝后的精度：', new_accuracy * 100, '%')
                print('    故需要剪枝！')
            else:
                nodes[i].set_is_leaf(False)
                nodes[i].set_value(original_value)
                print('    剪枝前的精度', original_accuracy * 100, '%', '大于等于', '剪枝后的精度：', new_accuracy * 100, '%')
                print('    故不需要剪枝！')
            print()
        i = i - 1
    return calculate_accuracy(D_for_test, root)


def deal_with_continuous_attributes(D, A):
    # 对属性集的属性循环找到连续属性
    for key, val in A.items():
        # 如果是连续属性
        if isinstance(val[0], int) or isinstance(val[0], float):
            # 得到连续属性的所有不等的属性值排序后的列表
            list_of_continuous_values = sorted(val)
            # 将属性集的属性值变为0和1，0表示小于划分点的属性值，1为大于划分点的属性值
            A[key] = [0, 1]
            max_gain = 0
            # tmp_D用于计算最优的划分点
            tmp_D = copy.deepcopy(D)
            # copy_of_D用于保存D最开始的数据
            copy_of_D = copy.deepcopy(D)
            # 开始按不同的划分点逐一划分
            for v in list_of_continuous_values[0: -1]:
                for i in range(0, len(D)):
                    if copy_of_D[i][0][key] > v:
                        tmp_D[i][0][key] = 1
                    else:
                        tmp_D[i][0][key] = 0
                # 找到信息增益最大的划分
                tmp_gain = information_gain(tmp_D, key)
                if tmp_gain > max_gain:
                    max_gain = tmp_gain
                    for i in range(0, len(D)):
                        if copy_of_D[i][0][key] > v:
                            D[i][0][key] = 1
                        else:
                            D[i][0][key] = 0


# 读鸢尾花数据集文件
D_and_A = read_file("/Users/mac/Downloads/iris.xlsx")
D = np.array(D_and_A[0])
A = D_and_A[1]
# 处理连续值
deal_with_continuous_attributes(D, A)
avg_for_not_pruning = 0.
avg_for_post_pruning = 0.
# 随机取样训练测试100次，统计平均精度
for i in range(100):
    # 共有150个样本，从0到149中随机选取100个不重复的数，对应的100个样本作为训练集
    random_100_samples_for_train = random.sample(range(0, 150), 100)
    # 剩下的50个随机的数，对应的50个样本作为测试集
    random_50_samples_for_test = \
        list(set(range(0, 150)).difference(set(random_100_samples_for_train)))
    # 生成训练集
    D_for_train = D[random_100_samples_for_train, :]
    # 生成测试集
    D_for_test = D[random_50_samples_for_test, :]
    # 用训练集训练生成决策树
    decision_tree_root = aTreeGenerate(D_for_train, A)
    print('@不剪枝的分类器：')
    avg_for_not_pruning = \
        avg_for_not_pruning + calculate_accuracy(D_for_test, decision_tree_root)
    print()
    print('@后剪枝的分类器：')
    avg_for_post_pruning = \
        avg_for_post_pruning + post_pruning(decision_tree_root, D_for_test)
print()
print('随机训练测试100次鸢尾花数据集，所得的平均精度如下：')
print('   鸢尾花数据集不剪枝平均精度：', avg_for_not_pruning, '%')
print('   鸢尾花数据集后剪枝平均精度：', avg_for_post_pruning, '%')
