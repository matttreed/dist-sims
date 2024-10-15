# Distributed Training Simulator

There are now two simulators for distributed training: WASH-Sim and Loco-Sim.

## WASH-Sim

WASH-Sim is a simulator for the WASH algorithm, which is a distributed training algorithm that intends to minimize latency in model training due to low connectivity between devices.

WASH-Sim merely simulates this distributed network, but all workers are running on the same machine.

Example usage can be found in the `examples` directory.

```python

from wash import WashingMachine
from models import ModelArchitecture
import torch.nn.functional as F
from data import train_dataset, test_dataset

wm = WashingMachine(
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
)

wm.load_model("chkpts/model_chkpt.pth")

wm.train()

```



## Loco-Sim

Loco-Sim is a simulator for the DiLoCo algorithm, which is a distributed training algorithm that synchronizes models every n steps instead of every step.

Loco-Sim merely simulates this distributed network, but all workers are running on the same machine.

Example usage can be found in the `examples` directory.

```python

from diloco import LocoMachine
from models import ModelArchitecture
import torch.nn.functional as F
from data import train_dataset, test_dataset

lc = LocoMachine(
    model_cls=ModelArchitecture,
    model_kwargs={
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_classes": 10,
        ...
    },
    optimizer_cls=torch.optim.AdamW,
    optimizer_kwargs={"lr": 0.01},
    outer_optimizer_cls=torch.optim.SGD,
    outer_optimizer_kwargs={"lr": 0.01, "nesterov": True, "momentum": 0.9},
    dataloader_kwargs={},
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=F.cross_entropy,
    num_workers=4,
    num_epochs=10,
    sync_interval=8,
    batch_size=16,
    save_path="outputs/final_model.pth",
)

lc.load_model("chkpts/model_chkpt.pth")

lc.train()

```