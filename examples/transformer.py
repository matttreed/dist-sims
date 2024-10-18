import torch
import torch.nn.functional as F
from wash import WashingMachine
from tinygpt import GPTConfig, GPT
from data import TextDataset


def CELoss(inputs, targets):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))


if __name__ == "__main__":

    gptconf = GPTConfig(
        block_size=256, vocab_size=50304, n_layer=2, n_head=4, n_embd=128
    )

    train_dataset = TextDataset(
        "data/tinystories/tinystories_tokenized_with_eot.bin",
        seq_length=gptconf.block_size,
    )

    wm = WashingMachine(
        model_cls=GPT,
        model_kwargs={
            "config": gptconf,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.01},
        dataloader_kwargs={},
        train_dataset=train_dataset,
        eval_dataset=None,
        loss_fn=CELoss,
        num_workers=2,
        num_epochs=1,
        shuffle_interval=1,
        p_shuffle=0.01,
        batch_size=16,
        split_dataset=True,
        modulate_p_shuffle=True,
        save_path="outputs/transformer.pth",
    )

    wm.train()
