import TEST_1
import feature_extraction_1
import filter

#test_file_path = "./LabeledDB_select/"
test_file_path = "./instance38.xls"

database = "./database_feature_v5_test.xls"

database_feature_list = TEST_1.excel_read(database)

#mesh_feature_list = feature_extraction_1.feature_extract(test_file_path)
mesh_feature_list = TEST_1.excel_read(test_file_path)

recall_list = []
c = 5
output_list = []

for  mesh_feature in mesh_feature_list:

    result_list = TEST_1.cal_similarity(mesh_feature, database_feature_list)
    result_list.sort(key=TEST_1.by_3rd)
    #c = len( list(x for x in result_list if x[0] == mesh_feature[0]) )
    #accuracy = TEST.cal_accuracy(mesh_feature[0], c, list(x[0] for x in result_list))
    recall = TEST_1.cal_recall(mesh_feature[0], c, list(x[0] for x in result_list))

    recall_list.append(recall)
    #print('Shape ',mesh_feature[1], ' Accuracy: ', accuracy)
    print('Shape ',mesh_feature[1], ' Recall: ', recall)

    #excel output data
    temp = [mesh_feature[0], str( list(x[0] for x in result_list[0:5]) ), recall]
    output_list.append(temp)

print('AVG: ',sum(recall_list)/len(recall_list))

workbook_output = filter.excel_parse(output_list, title=["type","result","precision"])

workbook_output.save("Edu_result.xls")