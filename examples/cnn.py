import torch
from wash import WashingMachine
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from examples.models import CNNModel
import torch.nn.functional as F


if __name__ == "__main__":

    torch.manual_seed(12345)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

    wm = WashingMachine(
        model_cls=CNNModel,
        model_kwargs={
            "input_channels": 1,
            "input_height": 28,
            "input_width": 28,
            "num_classes": 10,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.005},
        dataloader_kwargs={},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=F.cross_entropy,
        num_workers=10,
        num_epochs=4,
        shuffle_interval=1,
        p_shuffle=0.01,
        batch_size=32,
        split_dataset=True,
        save_path="outputs/cnn_model.pth",
    )

    wm.load_model("outputs/cnn_model.pth")

    wm.train()
