from sklearn.manifold import TSNE
import TEST
import numpy as np
import matplotlib.pyplot as plt

def plot_embedding(data, label, name , title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], label[i]+name[i],
                 color=plt.cm.Set1( (i/16) / 10.),
                 fontdict={'weight': 'bold', 'size': 6})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return fig

database_feature_list = TEST.excel_read("./database_feature_v3_test.xls")

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

fig = plot_embedding(result, y, name, 'tsne result')






