from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import TEST
import scipy

#read dataset
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
#test k=1 to 31
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 5:1 seperate
    scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
    k_error.append(1 - scores.mean())

#figure draw
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()
