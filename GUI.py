import PySimpleGUI as sg
import trimesh
import feature_extraction_1
import TEST
import normalize
from trimesh.viewer import windowed
from trimesh import viewer
def openfile():
    sg.theme('DarkAmber')

    layout: list = [
        [
            sg.Text('Choose the input mesh:'),
            sg.VSeparator(),
            sg.Input(key='-SourceFileInput-', expand_x=True),
            sg.VSeparator(),
            sg.FileBrowse(key='-SourceFile-', target=(sg.ThisRow, -2))
        ],

        [sg.Button('Search'), sg.Button('Cancel')]

    ]

    file_path = []
    window = sg.Window('Shape search', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        if event == 'Search':
            file_path = values['-SourceFileInput-']
            break

    window.close()
    return file_path

def show_result(mesh_feature, mesh, result_list):
    sg.theme('DarkAmber')

    layout: list = [
        [   sg.Text('Original mesh class: ' + mesh_feature[0])],
        [   sg.Text('Original mesh: '),
            sg.Button('show', key = 'o1')    
        ],

        [   sg.Text('Quering result:')],

        [   
            sg.Text(result_list[0][1]),
            sg.Text('Distance: '+ str(format(result_list[0][2],'.3f'))),
            sg.Button('Show',key = 'b1')],
        [
            sg.Text(result_list[1][1]),
            sg.Text('Distance: '+ str(format(result_list[1][2],'.3f'))),
            sg.Button('Show',key = 'b2')],
        [
            sg.Text(result_list[2][1]),
            sg.Text('Distance: '+ str(format(result_list[2][2],'.3f'))),
            sg.Button('Show',key = 'b3')],
        [
            sg.Text(result_list[3][1]),
            sg.Text('Distance: '+ str(format(result_list[3][2],'.3f'))),
            sg.Button('Show',key = 'b4')],
        [
            sg.Text(result_list[4][1]),
            sg.Text('Distance: '+ str(format(result_list[4][2],'.3f'))),
            sg.Button('Show',key = 'b5')]
        
    ]

    window_2 = sg.Window('Search result', layout)

    while True:

        event_2, val_2 = window_2.read()

        if event_2 == sg.WIN_CLOSED:
            break
        if event_2 == 'o1':         
            scene = trimesh.scene.scene.Scene(mesh)
            trimesh.viewer.SceneViewer(scene, resolution=(1280,720))       
        if event_2 == 'b1':
            file_path = result_list[0][1]
            mesh = trimesh.load_mesh(file_path)
            mesh = normalize.shape_normalize(mesh)
            #mesh.show()            
            scene = trimesh.scene.scene.Scene(mesh)
            trimesh.viewer.SceneViewer(scene, resolution=(1280,720))
        if event_2 == 'b2':
            file_path = result_list[1][1]
            mesh = trimesh.load_mesh(file_path)
            mesh = normalize.shape_normalize(mesh)
            #mesh.show()            
            scene = trimesh.scene.scene.Scene(mesh)
            trimesh.viewer.SceneViewer(scene, resolution=(1280,720))
        if event_2 == 'b3':
            file_path = result_list[2][1]
            mesh = trimesh.load_mesh(file_path)
            mesh = normalize.shape_normalize(mesh)
            #mesh.show()
            scene = trimesh.scene.scene.Scene(mesh)
            trimesh.viewer.SceneViewer(scene, resolution=(1280,720))
        if event_2 == 'b4':
            file_path = result_list[3][1]
            mesh = trimesh.load_mesh(file_path)
            mesh = normalize.shape_normalize(mesh)
            #mesh.show()
            scene = trimesh.scene.scene.Scene(mesh)
            trimesh.viewer.SceneViewer(scene, resolution=(1280,720))
        if event_2 == 'b5':
            file_path = result_list[4][1]
            mesh = trimesh.load_mesh(file_path)
            mesh = normalize.shape_normalize(mesh)
            #mesh.show()
            scene = trimesh.scene.scene.Scene(mesh)
            trimesh.viewer.SceneViewer(scene, resolution=(1280,720))

    window_2.close()

    return 

file_path = openfile()

database = "./database_feature_v5_test.xls"

database_feature_list = TEST.excel_read(database)

mesh_feature, mesh = feature_extraction_1.feature_extract_single(file_path)

result_list = TEST.cal_similarity(mesh_feature, database_feature_list)

result_list.sort(key=TEST.by_3rd)

#scene = trimesh.scene.scene.Scene(mesh)
#trimesh.viewer.SceneViewer(scene, resolution=(1280,720))
#mesh.show()

show_result(mesh_feature, mesh, result_list[0:5])


