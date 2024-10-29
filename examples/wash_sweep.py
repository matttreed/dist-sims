import torch
import torch.nn.functional as F
from distsims import WashingMachine
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

    # gptconf = GPTConfig(
    #     block_size=256, vocab_size=50304, n_layer=4, n_head=8, n_embd=128
    # )

    train_dataset = TextDataset(
        "data/owt/openwebtext.bin",
        dtype=np.uint16,
        seq_length=gptconf.block_size,
    )

    for num_workers in [1, 4]:

        batch_size = 16

        # if num_workers == 0:
        #     num_workers = 1
        #     batch_size = 32

        wm = WashingMachine(
            model_cls=GPT,
            model_kwargs={
                "config": gptconf,
            },
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 0.0005},
            dataloader_kwargs={"shuffle": True},
            synchronize_method="wash",
            synchronize_interval=1,
            eval_interval=500,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            loss_fn=CELoss,
            num_workers=num_workers,
            num_epochs=1,
            p_shuffle=0.01,
            batch_size=batch_size,
            data_parallel=True,
            modulate_p_shuffle=False,
            save_dir=f"outputs/wash_sweep_{num_workers}_workers",
            wandb_project="wash-sweep",
            max_local_step=5000,
        )

        wm.load_model("outputs/transformer_4/ckpt_epoch_0_iter_10000.pth")

        wm.train()
