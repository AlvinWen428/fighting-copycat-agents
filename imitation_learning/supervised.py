import numpy as np
import torch

from imitation_learning.utils.misc import no_grad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@no_grad()
def test(mode, model, test_dataloader, criterion):
    """Compute test loss."""
    losses = []
    for it, batch in enumerate(test_dataloader):
        x, y = mode.batch(batch), batch.labels()
        x, y = x.to(device), y.to(device)

        current_action = y[:, -1]
        prev_action = y[:, -2]

        output = model(x)

        loss = criterion(output, current_action)
        losses.append(loss.item())
    return {"test_loss": np.mean(losses)}
