import torch
import torch.nn.functional as F
from dist import WashingMachine
from examples.nanogpt import GPTConfig, GPT
from data import TextDataset
import numpy as np


def CELoss(inputs, targets):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))


if __name__ == "__main__":

    # gptconf = GPTConfig(
    #     block_size=256, vocab_size=50304, n_layer=2, n_head=4, n_embd=128
    # )

    gptconf = GPTConfig(
        block_size=512, vocab_size=50304, n_layer=12, n_head=8, n_embd=512
    )

    train_dataset = TextDataset(
        "data/owt/openwebtext.bin",
        dtype=np.uint16,
        seq_length=gptconf.block_size,
    )

    wm = WashingMachine(
        model_cls=GPT,
        model_kwargs={
            "config": gptconf,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.0001},
        dataloader_kwargs={"shuffle": True},
        synchronize_method="diloco",
        synchronize_interval=20,
        outer_optimizer_cls=torch.optim.SGD,
        outer_optimizer_kwargs={"lr": 0.001, "nesterov": True, "momentum": 0.9},
        train_dataset=train_dataset,
        eval_dataset=None,
        loss_fn=CELoss,
        num_workers=2,
        num_epochs=1,
        ckpt_interval=2000,
        p_shuffle=0.01,
        batch_size=16,
        data_parallel=True,
        modulate_p_shuffle=True,
        save_dir="outputs/diloco_20",
        wandb_project="wash-transformer",
        drift_penalty=0.01,
    )

    wm.train()
