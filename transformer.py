import torch
import torch.nn.functional as F
from dist import WashingMachine
from nanogpt import GPTConfig, GPT
from data import TextDataset
import numpy as np
import argparse


def CELoss(inputs, targets):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--p_shuffle", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--outer_learning_rate", type=float, default=0.7)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--data_parallel", type=bool, default=True)
    parser.add_argument("--wash_interval", type=int, default=1)
    parser.add_argument("--max_local_step", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--synchronize_interval", type=int, default=None)
    parser.add_argument(
        "--synchronize_method", type=str, choices=["diloco", "avg"], default="avg"
    )
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--cosine_anneal", action="store_true")

    args = parser.parse_args()

    gptconf = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        n_embd=args.embed_size,
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
        optimizer_kwargs={"lr": args.learning_rate},
        outer_optimizer_cls=torch.optim.SGD,
        outer_optimizer_kwargs={
            "lr": args.outer_learning_rate,
            "nesterov": True,
            "momentum": 0.9,
        },
        synchronize_method=args.synchronize_method,
        synchronize_interval=args.synchronize_interval,
        wash_interval=args.wash_interval,
        eval_interval=args.eval_interval,
        train_dataset=train_dataset,
        loss_fn=CELoss,
        num_workers=args.num_workers,
        p_shuffle=args.p_shuffle,
        batch_size=args.batch_size,
        modulate_p_shuffle=False,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        max_local_step=args.max_local_step,
        eval_iters=args.eval_iters,
        cosine_anneal=args.cosine_anneal,
    )

    if args.model_path:
        wm.load_model(args.model_path)

    wm.train()
