import numpy as np
import trimesh
import xlwt
import os
import string

def find_all_shape_file(database):
    for root,dirs,files in os.walk(database):
        for file in files:
            if file.endswith('.off') or file.endswith('.ply'):
                fullname_list = os.path.join(root, file)
                yield fullname_list

def excel_parse(mesh_information_list,title):
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('1')
    
    for i,item in enumerate(title):
        worksheet.write(0,i,item)
    
    for i,item in enumerate(mesh_information_list):
        for j,data in enumerate(item):
            worksheet.write(i+1,j,data)
    
    return workbook


'''

#count = 0
database = "./LabeledDB_new/"
mesh_information_list = []
for a in find_all_shape_file(database):
    type = a[a.rindex("/")+1:a.rindex("\\")]
    mesh = trimesh.load_mesh(a)
    #tracked array
    mesh_information = [type,len(mesh.vertices),len(mesh.faces),
        mesh.bounding_box.primitive.extents[0],
        mesh.bounding_box.primitive.extents[1],
        mesh.bounding_box.primitive.extents[2]
        ]
    mesh_information_list.append(mesh_information)
    #count = count + 1
    #if count>10 :break

title = ["class","number of vertices","number of faces","bound on x-axis","bound on y-axis","bound on z-axis"]

workbook_output = excel_parse(mesh_information_list,title)

workbook_output.save("database.xls")

#print(mesh_information_list)

'''
