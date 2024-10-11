# WASH-Sim: A WASH Simulator for PyTorch

WASH-Sim is a simulator for the WASH algorithm, which is a distributed training algorithm that intends to minimize latency in model training due to low connectivity between devices.

WASH-Sim merely simulates this distributed network, but all workers are running on the same machine.

Example usage can be found in the `examples` directory.

```python

from wash import simulate_wash
from models import ModelArchitecture
import torch.nn.functional as F
from data import train_dataset, test_dataset

simulate_wash(
    model_cls=ModelArchitecture,
    model_kwargs={
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_classes": 10,
        ...
    },
    optimizer_cls=torch.optim.AdamW,
    optimizer_kwargs={"lr": 0.01},
    dataloader_kwargs={},
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=F.cross_entropy,
    num_workers=4,
    num_epochs=10,
    shuffle_interval=1,
    p_shuffle=0.01,
    batch_size=16,
    split_dataset=False,
    save_path="outputs/final_model.pth",
):

```