import numpy as np
import torch
def data():
    #Getting everthing
    e = []
    with open("mnist.txt",'r') as file:
        for line in file:
            stuff = line.strip().split(",")
            e.append(stuff)
    #seperating labels and data
    labels = []
    data_pre = []
    for i in e:
        labels.append(int(i[784]))
        data_pre.append(i[:784]) 
    #Normalization
    data = []
    for d in data_pre:
        nd = [int(i) / 255 for i in d]
        data.append(np.array(nd))
    #Numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    #Shuffling
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    #Splitting data
    num = round(len(data)*6/7)
    t_data = torch.tensor(data[num:], dtype=torch.float)
    t_labels = torch.tensor(labels[num:], dtype=torch.long)
    data = torch.tensor(data[:num], dtype=torch.float)
    labels = torch.tensor(labels[:num], dtype=torch.long)
    #Return data and labels as testing data and labels and training data and labels
    return data,labels,t_data,t_labels
