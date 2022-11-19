from audioop import avg
import re
import trimesh
import normalize
import numpy as np
import numba
import math
from random import sample
from random import choice
from numpy.linalg import det
import filter
import sys

A3_times = 1000
D1_times = 1000
D2_times = 1000
D3_times = 1000
D4_times = 1000

def cal_distance_4(mesh,times):
    distance_4_list = []
    for i in range(times):
        rvertice_3, face_index=trimesh.sample.sample_surface(mesh,4)
        n = np.insert(rvertice_3.T,0,values=[1,1,1,1],axis=0)
        volume = abs(det(n)/6)
        cube_root = pow(volume,1/3)
        distance_4_list.append(cube_root)
    return distance_4_list

def cal_distance_3(mesh,times):
    distance_3_list = []
    for i in range(times):
        sample, face_index=trimesh.sample.sample_surface(mesh,3)
        square_root = math.sqrt(trimesh.triangles.area(triangles=[sample],sum = True))
        distance_3_list.append(square_root)
    return distance_3_list

def cal_distance_2(mesh,times):
    distance_2_list = []
    for i in range(times):
        rvertice_2, face_index=trimesh.sample.sample_surface(mesh,2)
        distance_2 = np.sqrt( pow(rvertice_2[0][0]-rvertice_2[1][0],2) 
                            + pow(rvertice_2[0][1]-rvertice_2[1][1],2) 
                            + pow(rvertice_2[0][2]-rvertice_2[1][2],2))
        distance_2_list.append(distance_2)
    return distance_2_list

def cal_distance_1(mesh,times):
    distance_1_list = []
    for i in range(times):
        rvertice_1 = choice(mesh.vertices)
        distance_1 = np.sqrt( pow(mesh.center_mass[0]-rvertice_1[0],2) 
                            + pow(mesh.center_mass[1]-rvertice_1[1],2) 
                            + pow(mesh.center_mass[2]-rvertice_1[2],2))
        distance_1_list.append(distance_1)
    return distance_1_list

def cal_angle(mesh,times):
    angle_list = []
    for i in range(times):
        sample, face_index=trimesh.sample.sample_surface(mesh,3)
        angle = cal_angle_between_3_points(sample)
        angle_list.append(angle)
    return angle_list

@numba.jit(nopython=True)
def cal_diameter(vertices):
    max_distance = 0
    for i in range(len(vertices) - 1):
        for j in range(i + 1, len(vertices)):
            distance = np.sqrt(pow(vertices[i][0] - vertices[j][0], 2) + pow(
                vertices[i][1] - vertices[j][1], 2) + pow(
                vertices[i][2] - vertices[j][2], 2))
            if distance > max_distance:
                max_distance = distance
    return max_distance

@numba.jit(nopython=True)
def cal_angle_between_3_points(list_vertices):
    i = math.sqrt(pow(list_vertices[0][0]-list_vertices[1][0],2) 
                + pow(list_vertices[0][1]-list_vertices[1][1],2) 
                + pow(list_vertices[0][2]-list_vertices[1][2],2))
    j = math.sqrt(pow(list_vertices[0][0]-list_vertices[2][0],2) 
                + pow(list_vertices[0][1]-list_vertices[2][1],2) 
                + pow(list_vertices[0][2]-list_vertices[2][2],2))        
    t = math.sqrt(pow(list_vertices[1][0]-list_vertices[2][0],2) 
                + pow(list_vertices[1][1]-list_vertices[2][1],2) 
                + pow(list_vertices[1][2]-list_vertices[2][2],2))    
    
    a = math.degrees(math.acos((pow(i,2)-pow(j,2)-pow(t,2))/-2/j/t))
    b = math.degrees(math.acos((pow(j,2)-pow(i,2)-pow(t,2))/-2/i/t))
    c = math.degrees(math.acos((pow(t,2)-pow(i,2)-pow(j,2))/-2/i/j))

    return max(a,b,c)

def feature_extract(database):

    count = 1
    g = []
    mesh_feature_list = []
    area_avg = 1.202119818
    area_stdev = 0.873605804
    campactness_avg = 14.09013066
    campactness_stdev = 17.3149124
    rectangularity_avg = 0.188290255
    rectangularity_stdev = 0.154387105
    eccentricity_avg = 3.435254744
    eccentricity_stdev = 2.488860436
    diameter_avg = 1.057077307
    diameter_stdev = 0.07824286



    for a in filter.find_all_shape_file(database):
        print('start', count)
        sys.stdout.flush()

        type = a[a.rindex("/")+1:a.rindex("\\")]
        mesh = trimesh.load_mesh(a)

        #normalize
        mesh = normalize.shape_normalize(mesh)
        print('normalize finish',count)
        sys.stdout.flush()

        #feature
        area = mesh.area#surface area
        campactness = pow(mesh.area,3)/36/np.pi/pow(mesh.volume,2)#compactness
        box_volume = mesh.volume/(mesh.bounding_box.primitive.extents[0] * mesh.bounding_box.primitive.extents[1] * mesh.bounding_box.primitive.extents[2])#bounding-box volume
        rectangularity = abs(box_volume)
        eccentricity = max(mesh.bounding_box.primitive.extents)/min(mesh.bounding_box.primitive.extents)#eccentricity
        diameter = cal_diameter(mesh.vertices)


        #feature normalize
        area = abs((area - area_avg) / area_stdev)
        campactness = abs((area - campactness_avg) / campactness_stdev)
        #box_volume = abs((box_volume - box_volume_avg) / box_volume_stdev)
        rectangularity = abs((rectangularity - rectangularity_avg) / rectangularity_stdev)
        eccentricity = abs((eccentricity - eccentricity_avg) / eccentricity_stdev)
        diameter = abs((diameter - diameter_avg) / diameter_stdev)

        #feature should be in histogram 
        angle_list = cal_angle(mesh,A3_times)
        distance_1_list = cal_distance_1(mesh,D1_times)
        distance_2_list = cal_distance_2(mesh,D2_times)
        distance_3_list = cal_distance_3(mesh,D3_times)
        distance_4_list = cal_distance_4(mesh,D4_times)

        #normalize angle_list to 0-1
        new_angle_list = []
        for i in angle_list:
            new_angle_list.append((i - 60)/120)

        # normalize D1 to 0-1
        new_distance_1_list = []
        for i in distance_1_list:
            new_distance_1_list.append(i)

        # normalize D2 to 0-1
        new_distance_2_list = []
        for i in distance_2_list:
            new_distance_2_list.append(i)

        # normalize D3 to 0-1
        new_distance_3_list = []
        for i in distance_3_list:
            new_distance_3_list.append((i*100)/45)

        # normalize D4 to 0-1
        new_distance_4_list = []
        for i in distance_4_list:
            new_distance_4_list.append((i*20)/9)

        #histogram
        angle_frequency, angle_bin = np.histogram(new_angle_list, bins = 20, range=(0,1))
        d1_frequency, d1_bin = np.histogram(new_distance_1_list, bins = 20, range=(0,1))
        d2_frequency, d2_bin = np.histogram(new_distance_2_list, bins = 20, range=(0,1))
        d3_frequency, d3_bin = np.histogram(new_distance_3_list, bins = 20, range=(0,1))
        d4_frequency, d4_bin = np.histogram(new_distance_4_list, bins = 20, range=(0,1))

        path = a

        mesh_feature = [type, path, area, campactness, rectangularity, eccentricity, diameter
                        ]

        #temp_frequency = angle_frequency + d1_frequency + d2_frequency + d3_frequency + d4_frequency
        temp_frequency = []
        temp_frequency.extend(angle_frequency)
        temp_frequency.extend(d1_frequency)
        temp_frequency.extend(d2_frequency)
        temp_frequency.extend(d3_frequency)
        temp_frequency.extend(d4_frequency)

        for i in range(100):
            mesh_feature.append(int(temp_frequency[i]))

        mesh_feature_list.append(mesh_feature)
        
        print('finish', count,flush=True)

        count = count + 1
        #if count == 2:
        #   break

    return mesh_feature_list

def feature_extract_single(file_path):

    area_avg = 1.202119818
    area_stdev = 0.873605804
    campactness_avg = 14.09013066
    campactness_stdev = 17.3149124
    rectangularity_avg = 0.188290255
    rectangularity_stdev = 0.154387105
    eccentricity_avg = 3.435254744
    eccentricity_stdev = 2.488860436
    diameter_avg = 1.057077307
    diameter_stdev = 0.07824286


    last_seperater = file_path.rindex("/")
    before_last_seperater = file_path.rindex("/",0 , last_seperater-1)
    type = file_path[before_last_seperater + 1 : last_seperater]
    mesh = trimesh.load_mesh(file_path)

    #normalize

    print('Normalize in progress...')

    mesh = normalize.shape_normalize(mesh)

    print('Normalize finish')
    sys.stdout.flush()

    print('Feature extraction in progress...',flush=True)
    #feature
    area = mesh.area#surface area
    campactness = pow(mesh.area,3)/36/np.pi/pow(mesh.volume,2)#compactness
    box_volume = mesh.volume/(mesh.bounding_box.primitive.extents[0] * mesh.bounding_box.primitive.extents[1] * mesh.bounding_box.primitive.extents[2])#bounding-box volume
    rectangularity = abs(box_volume)
    eccentricity = max(mesh.bounding_box.primitive.extents)/min(mesh.bounding_box.primitive.extents)#eccentricity
    diameter = cal_diameter(mesh.vertices)


    #feature normalize
    area = abs((area - area_avg) / area_stdev)
    campactness = abs((area - campactness_avg) / campactness_stdev)
    #box_volume = abs((box_volume - box_volume_avg) / box_volume_stdev)
    rectangularity = abs((rectangularity - rectangularity_avg) / rectangularity_stdev)
    eccentricity = abs((eccentricity - eccentricity_avg) / eccentricity_stdev)
    diameter = abs((diameter - diameter_avg) / diameter_stdev)

    #feature should be in histogram 
    angle_list = cal_angle(mesh,A3_times)
    distance_1_list = cal_distance_1(mesh,D1_times)
    distance_2_list = cal_distance_2(mesh,D2_times)
    distance_3_list = cal_distance_3(mesh,D3_times)
    distance_4_list = cal_distance_4(mesh,D4_times)

    #normalize angle_list to 0-1
    new_angle_list = []
    for i in angle_list:
        new_angle_list.append((i - 60)/120)

    # normalize D1 to 0-1
    new_distance_1_list = []
    for i in distance_1_list:
        new_distance_1_list.append(i)

    # normalize D2 to 0-1
    new_distance_2_list = []
    for i in distance_2_list:
        new_distance_2_list.append(i)

    # normalize D3 to 0-1
    new_distance_3_list = []
    for i in distance_3_list:
        new_distance_3_list.append((i*100)/45)

    # normalize D4 to 0-1
    new_distance_4_list = []
    for i in distance_4_list:
        new_distance_4_list.append((i*20)/9)

    #histogram
    angle_frequency, angle_bin = np.histogram(new_angle_list, bins = 20, range=(0,1))
    d1_frequency, d1_bin = np.histogram(new_distance_1_list, bins = 20, range=(0,1))
    d2_frequency, d2_bin = np.histogram(new_distance_2_list, bins = 20, range=(0,1))
    d3_frequency, d3_bin = np.histogram(new_distance_3_list, bins = 20, range=(0,1))
    d4_frequency, d4_bin = np.histogram(new_distance_4_list, bins = 20, range=(0,1))

    path = file_path

    mesh_feature = [type, path, area, campactness, rectangularity, eccentricity, diameter
                    ]

    #temp_frequency = angle_frequency + d1_frequency + d2_frequency + d3_frequency + d4_frequency
    temp_frequency = []
    temp_frequency.extend(angle_frequency)
    temp_frequency.extend(d1_frequency)
    temp_frequency.extend(d2_frequency)
    temp_frequency.extend(d3_frequency)
    temp_frequency.extend(d4_frequency)

    for i in range(100):
        mesh_feature.append(int(temp_frequency[i]))
    
    print('Feature extraction finish',flush=True)

    return mesh_feature, mesh


'''
#database = GUI.openfile()
database = "./LabeledDB_test/"

title = ["type", "path", "area","campactness","rectangularity","eccentricity","diameter",
        "angle_frequency",
        "d1_frequency", 
        "d2_frequency", 
        "d3_frequency", 
        "d4_frequency"]

mesh_feature_list = feature_extract(database)

workbook_output = filter.excel_parse(mesh_feature_list,title)

workbook_output.save("instance38.xls")

'''


