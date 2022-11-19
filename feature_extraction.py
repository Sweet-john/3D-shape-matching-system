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

    count = 0
    mesh_feature_list = []
    area_avg = 0.690514433
    area_stdev = 0.397404106
    campactness_avg = 14.09013066
    campactness_stdev = 17.3149124
    box_volume_avg = 0.125449541
    box_volume_stdev = 0.063771274
    eccentricity_avg = 3.435254744
    eccentricity_stdev = 2.488860436
    diameter_avg = 0.834427146
    diameter_stdev = 0.079850935



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
        box_volume = mesh.bounding_box.primitive.extents[0] * mesh.bounding_box.primitive.extents[1] * mesh.bounding_box.primitive.extents[2]#bounding-box volume
        eccentricity = max(mesh.bounding_box.primitive.extents)/min(mesh.bounding_box.primitive.extents)#eccentricity
        diameter = cal_diameter(mesh.vertices)

        #feature normalize
        area = abs((area - area_avg) / area_stdev)
        campactness = abs((area - campactness_avg) / campactness_stdev)
        box_volume = abs((box_volume - box_volume_avg) / box_volume_stdev)
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

        #histogram
        angle_frequency, angle_bin = np.histogram(new_angle_list, bins = 20, range=(0,1))
        d1_frequency, d1_bin = np.histogram(distance_1_list, bins = 20, range=(0,1))
        d2_frequency, d2_bin = np.histogram(distance_2_list, bins = 20, range=(0,1))
        d3_frequency, d3_bin = np.histogram(distance_3_list, bins = 20, range=(0,1))
        d4_frequency, d4_bin = np.histogram(distance_4_list, bins = 20, range=(0,1))

        mesh_feature = [type, area, campactness, box_volume, eccentricity, diameter,
        str(angle_frequency),
        str(d1_frequency),
        str(d2_frequency),
        str(d3_frequency),
        str(d4_frequency)]

        mesh_feature_list.append(mesh_feature)
        
        print('finish', count,flush=True)

        count = count + 1
        #if count == 1:
        #   break

    return mesh_feature_list


'''

database = "./LabeledDB_new/"

title = ["type", "area","campactness","box_volume","eccentricity","diameter",
        "angle_frequency",
        "d1_frequency", 
        "d2_frequency", 
        "d3_frequency", 
        "d4_frequency"]

mesh_feature_list = feature_extract(database)

workbook_output = filter.excel_parse(mesh_feature_list,title)

workbook_output.save("database_feature2.xls")


'''

