import re
from tkinter.messagebox import RETRY
import numpy as np
from sklearn.decomposition import PCA
import trimesh
import filter

def rotate(mesh,axis):
    if axis == 0:
        mesh.vertices*=[-1,1,1]
    elif axis == 1:
        mesh.vertices*=[1,-1,1]
    elif axis == 2:
        mesh.vertices*=[1,1,-1]
    return mesh

def cal_if_rotate(mesh,axis):
    list_positive = []
    list_negative = []
    smallest_on_axis = 10

    for i in mesh.vertices:
        if i[axis] < smallest_on_axis:
            smallest_on_axis = i[axis]
    middle_on_axis = smallest_on_axis + (mesh.bounding_box.primitive.extents[axis]/2)

    for i,item in enumerate(mesh.vertices):

        if item[axis]>=middle_on_axis:
            list_positive.append(np.sqrt(pow(item[0],2) + pow(item[1],2) + pow(item[2],2)))
        else:
            list_negative.append(np.sqrt(pow(item[0],2) + pow(item[1],2) + pow(item[2],2)))

    #rotate depend on count
    if  len(list_positive) > len(list_negative):
        mesh = rotate(mesh,axis)

    return mesh

def moment_test(mesh):

    if mesh.moment_inertia[1][2] < 0:
            mesh = rotate(mesh,0)
    if mesh.moment_inertia[0][2] < 0:
            mesh = rotate(mesh,1)
    if mesh.moment_inertia[0][1] < 0:
            mesh = rotate(mesh,2)

    return mesh

def shape_remesh(mesh):
    new_vertices,new_faces = trimesh.remesh.subdivide(mesh.vertices,mesh.faces)
    new_mesh = trimesh.Trimesh(vertices=new_vertices,faces=new_faces)
    return new_mesh

def shape_normalize(mesh):
    #translation
    mesh.vertices -= mesh.center_mass

    
    #boundary = mesh.bounding_box.primitive.extents
    #diagonal = np.sqrt(pow(boundary[0],2)+pow(boundary[1],2)+pow(boundary[2],2))
    #mesh.vertices = mesh.vertices/diagonal

    #remesh
    while len(mesh.vertices)<6000:
        mesh = shape_remesh(mesh)

    #pca
    pca=PCA(n_components=3)
    pca.fit(mesh.vertices)
    #print('after pca:',len(pca.transform(mesh.vertices)))
    mesh.vertices= pca.transform(mesh.vertices)

    #scaling
    boundary = mesh.bounding_box.primitive.extents
    bmax = max(boundary)
    mesh.vertices = mesh.vertices/bmax

    #diagonal = np.sqrt(pow(boundary[0],2)+pow(boundary[1],2)+pow(boundary[2],2))
    #boundary_new = mesh.bounding_box.primitive.extents
    
    #flip
    #new_mesh = cal_if_rotate(mesh = mesh, axis = 0)
    #new_mesh = cal_if_rotate(mesh = mesh, axis = 1)
    #new_mesh = cal_if_rotate(mesh = mesh, axis = 2)

    #flip by moment test
    mesh = moment_test(mesh)

    return mesh

'''
database = "./LabeledDB_new/"
mesh_information_list = []
for a in filter.find_all_shape_file(database):
    type = a[a.rindex("/")+1:a.rindex("\\")]
    mesh = trimesh.load_mesh(a)

    #normalize
    mesh = shape_normalize(mesh)

    #tracked array
    mesh_information = [type,len(mesh.vertices),len(mesh.faces),
        mesh.bounding_box.primitive.extents[0],
        mesh.bounding_box.primitive.extents[1],
        mesh.bounding_box.primitive.extents[2]
        ]
    mesh_information_list.append(mesh_information)
    #count = count + 1
    #if count>10 :break

workbook_output = filter.excel_parse(mesh_information_list)

workbook_output.save("database_normalized3.xls")
'''