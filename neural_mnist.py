import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import mnist_data
print("Loading and organizing data")
data,labels,t_data,t_labels = mnist_data.data()
print("Data loaded and organized")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.network(x)
model = Net()
model.load_state_dict(torch.load(r"C:\Users\yanlo\Documents\mnist_best_model.pth"))
output = model(data)
while True:
    try:
        index = int(input("Enter an index:"))
        plt.imshow(data[index].reshape(28,28), cmap='Greys')
        plt.title(f'Prediction: {torch.argmax(output[index])}')
        plt.show()
    except:
        break