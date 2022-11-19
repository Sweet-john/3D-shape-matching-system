from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import TEST
import scipy

#读取数据集
database_feature_list = TEST.excel_read("./database_feature_v5_test.xls")
x = []
y = []
instance_list = [50]*20
for line in database_feature_list:
    '''
    temp = [] 
    temp.append(line[2:7])
    temp.append(line[7:27])
    temp.append(line[27:47])
    temp.append(line[47:67])
    temp.append(line[67:87])
    temp.append(line[87:107])
    '''

    distance_angle = scipy.stats.wasserstein_distance(instance_list, line[7:27])
    distance_d1 = scipy.stats.wasserstein_distance(instance_list, line[27:47])
    distance_d2 = scipy.stats.wasserstein_distance(instance_list, line[47:67])
    distance_d3 = scipy.stats.wasserstein_distance(instance_list, line[67:87])
    distance_d4 = scipy.stats.wasserstein_distance(instance_list, line[87:107])


    temp = []
    temp.extend(line[2:7])
    temp.extend([distance_angle,distance_d1,distance_d2,distance_d3,distance_d4])

    x.append(line[2:107])
    #x.append(temp)
    y.append(line[0])



k_range = range(1, 31)
k_error = []
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
    scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
    k_error.append(1 - scores.mean())

#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()
