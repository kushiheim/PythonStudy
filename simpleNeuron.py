import numpy as np
import matplotlib.pylab as plt


# パーセプトロンの実装例 : OR回路
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1.0, 1.0])
    b = -0.5
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))


# ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=int)


# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU関数,ランプ関数
def relu(x):
    return np.maximum(0, x)


# softmax
def softmax1(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


x = np.arange(-5.0, 5.0, 0.1)
y_step = step_function(x)
print(y_step)
y_sigmoid = sigmoid(x)
print(y_sigmoid)
y_relu = relu(x)
print(y_relu)
plt.plot(x, y_step, label="Step")
plt.plot(x, y_sigmoid, label="Sigmoid")
plt.plot(x, y_relu, label="ReLU")
plt.legend()
plt.ylim(-0.1, 1.5)
plt.grid(which='major', color='black', linestyle='--')
plt.grid(which='minor', color='black', linestyle='--')
plt.show()

# 行列計算
x = np.array([1, 2])
y = np.array([[3, 4], [5, 6]])
ans = np.dot(x, y)
print(ans)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([1010, 1000, 990])
print(softmax(a))
a = np.array([0.3, 4.0, 2.9])
print(softmax(a))
