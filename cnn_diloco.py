import torch
from diloco import LocoMachine
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR100
from models import CNNModel
import torch.nn.functional as F


if __name__ == "__main__":

    torch.manual_seed(12345)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    # test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

    train_dataset = CIFAR100(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = CIFAR100(
        root="./data", train=False, transform=transform, download=True
    )

    wm = LocoMachine(
        model_cls=CNNModel,
        model_kwargs={
            "input_channels": 3,
            "input_height": 32,
            "input_width": 32,
            "num_classes": 100,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.005},
        outer_optimizer_cls=torch.optim.SGD,
        outer_optimizer_kwargs={"lr": 0.01, "nesterov": True, "momentum": 0.9},
        dataloader_kwargs={},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=F.cross_entropy,
        num_workers=4,
        num_epochs=4,
        sync_interval=4,
        batch_size=8,
        save_path="outputs/cnn_model.pth",
    )

    wm.load_model("outputs/cnn_model.pth")

    wm.train()
