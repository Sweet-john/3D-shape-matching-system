from sklearn import neighbors
import TEST
import feature_extraction_1
import joblib
import filter
from sklearn.manifold import TSNE
import numpy as np

test_file_path = "./instance38.xls"

database_feature_list = TEST.excel_read("./database_feature_v5_test.xls")

mesh_feature_list = TEST.excel_read(test_file_path)

x=[]
y=[]
name = []
for line in database_feature_list:
    x.append(line[2:107])
    y.append(line[0])
    temp = line[1][line[1].rindex("\\")+1:]
    name.append(temp)

tsne = TSNE(n_components=2, perplexity=17, learning_rate='auto', n_iter= 100000)

result = tsne.fit_transform(np.array(x))

x=[]

for line in result:

    x.append(line)

Model = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')

Model.fit(x,y)

joblib.dump(Model,'./knn_tsne_1')

#-------------------


dis,index = Model.kneighbors( list (i[2:107] for i in mesh_feature_list) )

#result_path_list = list(path_list[i] for i in index[0])

T = 0
output_list = []

for num_x, item in enumerate(index):
    result_type_list = list(y[i] for i in item)

    #excel output set
    recall = TEST.cal_recall(mesh_feature_list[num_x][0], 5, result_type_list)
    output_list.append([mesh_feature_list[num_x][0], str(result_type_list) , recall])

    print('Num:',num_x)
    print(mesh_feature_list[num_x][0])
    print(result_type_list,'\n')

    for j in result_type_list:
        if j == mesh_feature_list[num_x][0]:
            T += 1


avg = T / len(mesh_feature_list) / 5

print('avg: ',avg)

print('Score:', Model.score(list (i[2:107] for i in mesh_feature_list) , list (i[0] for i in mesh_feature_list)))

workbook_output = filter.excel_parse(output_list, title=["type","result","precision"])

workbook_output.save("KNN_result_6.xls")
