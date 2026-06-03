import pathlib
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from rich import print as rprint
import torch
from torch import nn, optim

from interactive import interactive_mode
from mnist_data import data as get_data
from net import Net

if TYPE_CHECKING:
    from torch import Tensor

data, labels, t_data, t_labels = get_data()
model = Net()
lr = 0.005
epochs = input('Enter number of epochs: ')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
accuracies = []
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
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    accuracies.append(accuracy * 100)

    output = model(t_data)
    correct = 0
    _, predicted = torch.max(output, 1)
    correct = (predicted == t_labels).sum().item()
    t_accuracy = correct / t_labels.size(0)
    t_accuracies.append(t_accuracy * 100)
    print(
        f'Epoch {epoch:03}/{epochs:03}, Loss: {loss.item():.5f}, Accuracy: {accuracy:09.5%}, Test Accuracy: {t_accuracy:09.5%}'
    )

rprint('[bold green]Training complete[/bold green]')

output: Tensor = model(t_data)
correct = 0
_, predicted = torch.max(output, 1)
correct = (predicted == t_labels).sum().item()
t_accuracy = correct / t_labels.size(0)
a_acc = (t_accuracy + accuracy * 6) / 7 * 100

rprint(f'[blue bold]Test Accuracy: {t_accuracy * 100}[/blue bold]')
rprint(f'[blue bold]Average Accuracy: {a_acc}[/blue bold]')

try:
    with pathlib.Path('mnist_best_model_accuracy.txt').open(encoding='utf8') as file:
        best = float(file.read())
except FileNotFoundError:
    best = 0.0

if a_acc > best:
    with pathlib.Path('mnist_best_model_accuracy.txt').open('w', encoding='utf8') as file:
        file.write(str(a_acc))
    torch.save(model.state_dict(), 'mnist_best_model.pth')

plt.plot(range(len(accuracies)), accuracies, label='Training Accuracy')
plt.plot(range(len(t_accuracies)), t_accuracies, label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over time')
plt.show()

rprint('[bold]Entering interactive mode for test data...[bold]')
interactive_mode(t_data, output)
