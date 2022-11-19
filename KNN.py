from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
import TEST
import feature_extraction_1
import scipy

test_file_path = "./single_test/"

instance_type = 'FourLeg'

database_feature_list = TEST.excel_read("./database_feature_v3_test.xls")

mesh_feature_list = feature_extraction_1.feature_extract(test_file_path)

mesh_feature = mesh_feature_list[0]

x = []
y = []
path_list = []
for line in database_feature_list:

    x.append(line[2:107])
    y.append(line[0])
    path_list.append(line[1])

Model = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')

Model.fit(x,y)

dis,index = Model.kneighbors([mesh_feature[2:107]])
dis,index = Model.kneighbors([mesh_feature[2:107]])
result_path_list = list(path_list[i] for i in index[0])
print('distance:',dis, 'result path:',result_path_list)

print(Model.predict([mesh_feature[2:107]]))