import mnist_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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
lr = 0.005
epochs = int(input("Enter number of epochs:"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
acuracies = []
t_accuracies = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    acuracies.append(accuracy*100)

    output = model(t_data)
    correct = 0
    _, predicted = torch.max(output, 1)
    correct = (predicted == t_labels).sum().item()
    t_accuracy = correct / t_labels.size(0)
    t_accuracies.append(t_accuracy*100)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy*100}%')
print("Training complete")
output = model(t_data)
correct = 0
_, predicted = torch.max(output, 1)
correct = (predicted == t_labels).sum().item()
t_accuracy = correct / t_labels.size(0)
a_acc = (t_accuracy*100+accuracy*100)/2
print(f'Test Accuracy: {t_accuracy*100}')
print(f'Average Accuracy: {a_acc}')
with open(r"mnist_best_model_accuracy.txt",'r') as file:
    best = float(file.read())
if a_acc > best:
    with open(r"mnist_best_model_accuracy.txt",'w') as file:
        file.write(str(a_acc))
    torch.save(model.state_dict(), r"mnist_best_model.pth")
plt.plot(range(len(acuracies)),acuracies, label = 'Training Accuracy')
plt.plot(range(len(t_accuracies)),t_accuracies, label = 'Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over time')
plt.show()
while True:
    try:
        index = int(input("Enter an index:"))
        plt.imshow(t_data[index].reshape(28,28), cmap='Greys')
        plt.title(f'Prediction: {torch.argmax(output[index])}')
        plt.show()
    except:
        break
