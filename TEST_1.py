from unittest import result
import feature_extraction_1
import xlrd
import scipy
from scipy.spatial import distance

def excel_read(database):

    database_feature_list = []

    workbook = xlrd.open_workbook(database)
    table = workbook.sheet_by_index(0)

    for i in range(table.nrows):

        #jump title row
        if i == 0:
            continue

        feature_list = []

        for j in range(table.ncols):
            data = table.cell(i,j).value
            feature_list.append(data)

        database_feature_list.append(feature_list)

    return database_feature_list

def cal_accuracy(instance_type, c, result_type_list):

    TP = 0
    d = 342
    result_vector_len = c

    for i in range(result_vector_len):
        if instance_type == result_type_list[i]:
            TP += 1

    FP = result_vector_len - TP
    TN = d - c - FP
    accuracy = (TP + TN) / d

    return accuracy

def cal_recall(instance_type, c, result_type_list):

    TP = 0
    result_vector_len = c

    for i in range(result_vector_len):
        if instance_type == result_type_list[i]:
            TP += 1
    
    recall = TP / c

    return recall

def by_3rd(item):

    return item[2]

def cal_similarity(mesh_feature, database_feature_list):

    count = 0
    result_list = []

    #single feature
    instance_area = mesh_feature[2]
    instance_campactness = mesh_feature[3]
    instance_box_volume = mesh_feature[4]
    instance_eccentricity = mesh_feature[5]
    instance_diameter = mesh_feature[6]

    instance_global_list = list(i for i in mesh_feature[2:7])

    #histogram feature
    instance_angle_list = list(i for i in mesh_feature[7:27])
    instance_d1_list = list(i for i in mesh_feature[27:47])
    instance_d2_list = list(i for i in mesh_feature[47:67])
    instance_d3_list = list(i for i in mesh_feature[67:87])
    instance_d4_list = list(i for i in mesh_feature[87:107])

    for item in database_feature_list:
        
        item_area = item[2]
        item_campactness = item[3]
        item_box_volume = item[4]
        item_eccentricity = item[5]
        item_diameter = item[6]

        item_global_list = list(i for i in item[2:7])

        angle_list = list(i for i in item[7:27])
        d1_list = list(i for i in item[27:47])
        d2_list = list(i for i in item[47:67])
        d3_list = list(i for i in item[67:87])
        d4_list = list(i for i in item[87:107])
        '''
        distance_area = abs(item_area - instance_area)
        distance_campactness = abs(item_campactness - instance_campactness)
        distance_box_volume = abs(item_box_volume - instance_box_volume)
        distance_eccentricity = abs(item_eccentricity - instance_eccentricity)
        distance_diameter = abs(item_diameter - instance_diameter)
        '''

        global_distance = distance.euclidean(instance_global_list,item_global_list)

        distance_angle = distance.euclidean(instance_angle_list, angle_list)
        distance_d1 = distance.euclidean(instance_d1_list, d1_list)
        distance_d2 = distance.euclidean(instance_d2_list, d2_list)
        distance_d3 = distance.euclidean(instance_d3_list, d3_list)
        distance_d4 = distance.euclidean(instance_d4_list, d4_list)

        temp_compare = []

        temp_compare.append(3.3*global_distance)
        #print(global_distance)
        '''
        temp_compare.append(distance_area)
        temp_compare.append(distance_campactness)
        temp_compare.append(distance_box_volume)
        temp_compare.append(distance_eccentricity)
        temp_compare.append(distance_diameter)
        '''
        temp_compare.append(0.0005* distance_angle)
        temp_compare.append(0.0002* distance_d1)
        temp_compare.append(0.0005* distance_d2)
        temp_compare.append(0.0005* distance_d3)
        temp_compare.append(0.0005* distance_d4)
        
        result_list.append([item[0],item[1], sum(temp_compare), global_distance, distance_angle, distance_d1, distance_d2, distance_d3, distance_d4])

        count = count + 1
        #if count == 1:
        #    return
    return result_list


'''

#should be a folder
test_file_path = "./LabeledDB_select/single_test/"

instance_type = 'FourLeg'

database = "./database_feature_v21.xls"

database_feature_list = excel_read(database)

mesh_feature_list = feature_extraction_1.feature_extract(test_file_path)

#single test
mesh_feature = mesh_feature_list[0]

result_list = cal_similarity(mesh_feature, database_feature_list)

result_list.sort(key=by_3rd)

# number of relevant items in database

c = len( list(x for x in result_list if x[0] == instance_type) )

for i in range(c):
    print(result_list[i])

accuracy = cal_accuracy(instance_type, c, list(x[0] for x in result_list))
recall = cal_recall(instance_type, c, list(x[0] for x in result_list))

print('Accuracy: ', accuracy)
print('Recall: ', recall)

'''

