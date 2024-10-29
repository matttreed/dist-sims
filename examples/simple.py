import torch
import torch.random
from dist import WashingMachine
from torch.utils.data import Dataset
from examples.models import SimpleModel
import torch.nn.functional as F
import random


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
    random.seed(12345)

    train_dataset = SimpleDataset(torch.randn((2048, 2)), torch.randn((2048, 1)))
    test_dataset = SimpleDataset(torch.randn((2048, 2)), torch.randn((2048, 1)))

    wm = WashingMachine(
        model_cls=SimpleModel,
        model_kwargs={
            "input_size": 2,
            "hidden_size": 4,
            "output_size": 1,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.01},
        dataloader_kwargs={},
        synchronize_method="wash",
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=F.mse_loss,
        num_workers=2,
        num_epochs=1,
        synchronize_interval=1,
        p_shuffle=0.5,
        batch_size=16,
        data_parallel=False,
    )

    wm.train()
