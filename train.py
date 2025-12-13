import matplotlib.pyplot as plt
from rich import print as rprint
import torch
import torch.nn as nn
import torch.optim as optim

from interactive import interactive_mode
import mnist_data
from net import Net

rprint('[bold]Loading and organizing data...[/bold]')

data, labels, t_data, t_labels = mnist_data.data()

rprint('[bold green]Data loaded and organized.[/bold green]')

model = Net()
lr = 0.005
epochs = input('Enter number of epochs: ')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
acuracies = []
t_accuracies = []
try:
    epochs = int(epochs)
except ValueError:
    epochs = 100
    rprint('[red]Invalid input, defaulting to 100 epochs[/red]')
if epochs <= 0:
    rprint('[red]Number of epochs must be positive, defaulting to 100 epochs[/red]')
    epochs = 100

accuracy = 0
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    acuracies.append(accuracy * 100)

    output = model(t_data)
    correct = 0
    _, predicted = torch.max(output, 1)
    correct = (predicted == t_labels).sum().item()
    t_accuracy = correct / t_labels.size(0)
    t_accuracies.append(t_accuracy * 100)
    print(
        f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy * 100}%'
    )

rprint('[bold green]Training complete[/bold green]')

output = model(t_data)
correct = 0
_, predicted = torch.max(output, 1)
correct = (predicted == t_labels).sum().item()
t_accuracy = correct / t_labels.size(0)
a_acc = (t_accuracy + accuracy * 6) / 7 * 100

rprint(f'[blue bold]Test Accuracy: {t_accuracy * 100}[/blue bold]')
rprint(f'[blue bold]Average Accuracy: {a_acc}[/blue bold]')

try:
    with open('mnist_best_model_accuracy.txt') as file:
        best = float(file.read())
except FileNotFoundError:
    best = 0.0

if a_acc > best:
    with open('mnist_best_model_accuracy.txt', 'w') as file:
        file.write(str(a_acc))
    torch.save(model.state_dict(), 'mnist_best_model.pth')

plt.plot(range(len(acuracies)), acuracies, label='Training Accuracy')
plt.plot(range(len(t_accuracies)), t_accuracies, label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over time')
plt.show()

print('Entering interactive mode for test data...')
interactive_mode(t_data, output)
