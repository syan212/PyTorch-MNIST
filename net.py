from typing import TYPE_CHECKING, override

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10),
            nn.Softmax(dim=1),
        )
    
    @override
    def forward(self, x: Tensor):
        return self.network(x)
