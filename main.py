import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

N = 200

def get_random_normal(x_mean, x_dis, y_mean, y_dis):
    return [np.random.normal(x_mean, x_dis, size=N), np.random.normal(y_mean, y_dis, N)]

def distance(point_1, point_2):
    return math.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2)

def check(data_clasters, m):
    flag = True
    new_data = []
    for i in range(len(data_clasters)):
        new_data.append([])
    for i in range(len(data_clasters)):
        for elem in data_clasters[i]:
            min = i
            for j in range(len(m)):
                if i != j and distance(elem, m[j]) < distance(elem, m[i]):
                    flag = False
                    min = j
            new_data[min].append(elem)
    return new_data, flag


def k_average(data, start_clasters):
    data_clasters = []
    for i in range(len(start_clasters)):
        data_clasters.append([])
    for i in range(len(data)):
        min = 0
        for j in range(1, len(start_clasters)):
            if distance(data[i], start_clasters[j]) < distance(data[i], start_clasters[min]):
                min = j
        data_clasters[min].append(data[i])

    flag = False
    while not flag:
        m = []
        for i in range(len(data_clasters)):
            m.append([])
            arr_0 = []
            arr_1 = []
            for elem in data_clasters[i]:
                arr_0.append(elem[0])
                arr_1.append(elem[1])
            m[i] = [np.mean(arr_0), np.mean(arr_1)]
        data_clasters, flag = check(data_clasters, m)

    return data_clasters

def print_claster_data(data, color):
    x = []
    y = []
    for elem in data:
        x.append(elem[0])
        y.append(elem[1])
    plt.scatter(x, y, c=color)

claster1 = get_random_normal(3, 1.5, 3, 1.5)
claster2 = get_random_normal(9, 1, 2, 1)
claster3 = get_random_normal(9, 1, 6, 1)

#k - average
data = []
for i in range(N):
    data.append([claster1[0][i], claster1[1][i]])
for i in range(N):
    data.append([claster2[0][i], claster2[1][i]])
for i in range(N):
    data.append([claster3[0][i], claster3[1][i]])

answer = k_average(data, [[3, 3], [9, 2], [9, 6]])
colors = ['r', 'y', 'b']
for i in range(len(colors)):
    print_claster_data(answer[i], colors[i])
plt.legend(['(3, 3)', '(9, 2)', '(9, 6)'])
plt.show()
print()

def k_nearest(test, data):
    k = 5
    result = []
    for i in range(len(data)):
        result.append([])

    all_data = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            all_data.append([data[i][j], i])

    for i in range(len(test)):
        new_data = []
        for elem in all_data:
            new_data.append([distance(test[i], elem[0]), elem[1]])
        new_data.sort(key=lambda elem: elem[0])
        arr = []
        for j in range(k):
            arr.append(new_data[j][1])
        result[max(set(arr), key=arr.count)].append(test[i])

    return result

#k - nearest
print("Метод k ближайших")
test = [[3, 3], [9, 2], [9, 6], [5, 5], [1, 1], [10, 10], [1, 10], [10, 1], [10, 5], [5, 10]]
k_near = k_nearest(test, answer)
colors = ['r', 'y', 'b']
for i in range(len(colors)):
    print_claster_data(k_near[i], colors[i])
plt.legend(['(3, 3)', '(9, 2)', '(9, 6)'])
plt.show()
print('(3, 3):' + str(k_near[0]))
print('(9, 2):' + str(k_near[1]))
print('(9, 6):' + str(k_near[2]))
print()


def getTrainX(data):
    X = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            X.append(data[i][j])
    X = np.array(X)
    return X

def getTrainY(data):
    y = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            y.append(i)
    y = np.array(y)
    return y

#Naive Bayessovskiy method
print("Наивный Байсовский метод:")
test = [[3, 3], [9, 2], [9, 6], [5, 5], [1, 1], [10, 10], [1, 10], [10, 1], [10, 5], [5, 10]]
gnb = GaussianNB()
gnb.fit(getTrainX(answer).astype(float), getTrainY(answer).astype(int))
y_test = gnb.predict(test)
for i in range(len(test)):
    print(str(test[i]) + " : " + str(y_test[i]))
print()