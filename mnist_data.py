from typing import TYPE_CHECKING

import numpy as np
from rich import print as rprint
from rich.progress import track
import torch

if TYPE_CHECKING:
    from torch import Tensor


def data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rprint('[bold]Loading and organizing data...[/bold]')
    
    # Getting everthing
    e: list[list[str]] = []
    with open('mnist.txt') as file:
        for line in file:
            stuff = line.strip().split(',')
            e.append(stuff)
    with open('mnist2.txt') as file:
        for line in file:
            stuff = line.strip().split(',')
            e.append(stuff)

    # Seperating labels and data
    labels_pre: list[int] = []
    data_pre: list[list[str]] = []
    for i in track(e, description='Organizing data...'):
        labels_pre.append(int(i[784]))
        data_pre.append(i[:784])

    # Normalization
    data = []
    for d in track(data_pre, description='Normalizing data...'):
        nd = [int(i) / 255 for i in d]
        data.append(np.array(nd))

    # Numpy arrays
    data = np.array(data)
    labels = np.array(labels_pre)

    # Shuffling
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    # Splitting data
    num = round(len(data) * 6 / 7)
    t_data = torch.tensor(data[num:], dtype=torch.float)
    t_labels = torch.tensor(labels[num:], dtype=torch.long)
    data = torch.tensor(data[:num], dtype=torch.float)
    labels = torch.tensor(labels[:num], dtype=torch.long)

    rprint('[bold green]Data loaded and organized.[/bold green]')
    # Return data and labels as testing data and labels and training data and labels
    return data, labels, t_data, t_labels
