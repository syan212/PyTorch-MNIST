from rich import print as rprint
import torch

from interactive import interactive_mode
import mnist_data
from net import Net

data, labels, t_data, t_labels = mnist_data.data()

model = Net()
try:
    model.load_state_dict(torch.load('mnist_best_model.pth'))
    rprint('[green bold]Model loaded successfully[/green bold]')
except FileNotFoundError:
    rprint(
        '[red bold]No saved model found. Run train.py to train a model first.[/red bold]'
    )

all_data = torch.cat((t_data, data))
output = model(all_data)

rprint('[bold]Entering interactive mode for all data...[bold]')
interactive_mode(all_data, output)
