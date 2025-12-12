import matplotlib.pyplot as plt
import torch
import mnist_data
from net import Net

print("Loading and organizing data")

data, labels, t_data, t_labels = mnist_data.data()

print("Data loaded and organized")

model = Net()
model.load_state_dict(torch.load("mnist_best_model.pth"))
output = model(data)

while True:
    try:
        index = int(input("Enter an index:"))
        plt.imshow(data[index].reshape(28, 28), cmap="Greys")
        plt.title(f"Prediction: {torch.argmax(output[index])}")
        plt.show()
    except:
        break
