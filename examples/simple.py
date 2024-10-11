import torch
import torch.random
from wash import simulate_wash
from torch.utils.data import Dataset
from examples.models import SimpleModel
import torch.nn.functional as F


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super(SimpleDataset, self).__init__()
        assert x.size(0) == y.size(0)
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":

    torch.manual_seed(12345)

    train_dataset = SimpleDataset(torch.randn((64, 5)), torch.randn((64, 5)))
    test_dataset = SimpleDataset(torch.randn((64, 5)), torch.randn((64, 5)))

    simulate_wash(
        model_cls=SimpleModel,
        model_kwargs={
            "input_size": 5,
            "hidden_size": 10,
            "output_size": 5,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.01},
        dataloader_kwargs={},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=F.mse_loss,
        num_workers=4,
        num_epochs=10,
        shuffle_interval=1,
        p_shuffle=0.10,
        batch_size=16,
        split_dataset=True,
    )
