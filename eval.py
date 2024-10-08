import torch
from wash import wash_algorithm
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
from model import CNNModel
import torch.nn.functional as F


if __name__ == "__main__":

    torch.manual_seed(12345)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Normalize with mean and std for MNIST
        ]
    )

    # Load the MNIST dataset
    train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

    model = torch.load("final_model.pth")

    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16)

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Test accuracy: {correct / len(test_dataset)}")
