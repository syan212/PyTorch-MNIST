from sys import exit
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from rich import print as rprint
from rich.console import Console
import torch

if TYPE_CHECKING:
    from torch import Tensor


console = Console()


def interactive_mode(data: Tensor, output: Tensor) -> None:
    while True:
        try:
            index = input('Enter an index (enter q to exit): ')
            index = int(index)
            plt.imshow(data[index].reshape(28, 28), cmap='Greys')
            plt.title(f'Prediction: {torch.argmax(output[index])}')
            plt.show()
        except ValueError:
            if index == 'q':
                rprint('[red]Exiting[/red]')
                break
            if (
                console.input(
                    '[red]Invalid input.[/red] [bold]Do you want to exit? (y/n)[/bold]: '
                ).lower()
                == 'y'
            ):
                break
            rprint('[red]Please enter a valid integer index.')
        except IndexError:
            if (
                console.input(
                    '[red]Index out of range.[/red] [bold]Do you want to exit? (y/n)[/bold]: '
                ).lower()
                == 'y'
            ):
                break
            rprint(f'[red]Please enter an index between 0 and {data.size(0) - 1}.')
        except KeyboardInterrupt:
            rprint('\n[red]Exiting.')
            break
    exit()
