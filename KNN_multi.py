from sklearn import neighbors
import TEST
import feature_extraction_1
import joblib
import filter

#test_file_path = "./LabeledDB_select_1/"
test_file_path = "./instance38.xls"

database_feature_list = TEST.excel_read("./database_feature_v5_test.xls")

#mesh_feature_list = feature_extraction_1.feature_extract(test_file_path)
mesh_feature_list = TEST.excel_read(test_file_path)

x = []
y = []
path_list = []

for line in database_feature_list:

    x.append(line[2:107])
    y.append(line[0])
    path_list.append(line[1])

Model = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')

Model.fit(x,y)

joblib.dump(Model,'./knn_6')

#Model = joblib.load('./knn_1')

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