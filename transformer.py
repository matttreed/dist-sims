import torch
import torch.nn.functional as F
from dist import WashingMachine
from nanogpt import GPTConfig, GPT
from data import TextDataset
import numpy as np
import argparse
from util import arg_combinations, str2bool, generate_text
import random
import torch.autograd.profiler as profiler


def CELoss(inputs, targets):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", "-b", type=int, nargs="+", default=32)
    parser.add_argument("--num_workers", "-w", type=int, nargs="+", default=4)
    parser.add_argument("--p_shuffle", "-p", type=float, nargs="+", default=0.01)
    parser.add_argument("--learning_rate", "-lr", type=float, nargs="+", default=0.001)
    parser.add_argument("--outer_learning_rate", type=float, nargs="+", default=0.7)
    parser.add_argument("--data_parallel", type=bool, nargs="+", default=True)
    parser.add_argument("--wash_interval", type=int, nargs="+", default=1)
    parser.add_argument("--max_local_step", type=int, nargs="+", default=5000)
    parser.add_argument("--save_dir", type=str, nargs="+", default=None)
    parser.add_argument("--ckpt_interval", type=int, nargs="+", default=None)
    parser.add_argument("--model_path", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_project", type=str, nargs="+", default=None)
    parser.add_argument("--eval_interval", type=int, nargs="+", default=500)
    parser.add_argument("--eval_iters", type=int, nargs="+", default=100)
    parser.add_argument("--synchronize_interval", type=int, nargs="+", default=None)
    parser.add_argument("--synchronize_method", type=str, nargs="+", default="avg", choices=["avg", "diloco"])
    parser.add_argument("--shuffle_type", type=str, choices=["ring", "random"], nargs="+", default="random")
    parser.add_argument("--vocab_size", type=int, nargs="+", default=50304)
    parser.add_argument("--block_size", type=int, nargs="+", default=512)
    parser.add_argument("--num_layers", type=int, nargs="+", default=12)
    parser.add_argument("--num_heads", type=int, nargs="+", default=8)
    parser.add_argument("--embed_size", type=int, nargs="+", default=512)
    parser.add_argument("--cosine_anneal", type=str2bool, nargs="+", default=False)
    parser.add_argument("--modulate_p_shuffle", type=str2bool, nargs="+", default=False)
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--log_stats_interval", type=int, nargs="+", default=10)
    parser.add_argument("--drift_penalty", type=float, nargs="+", default=None)
    parser.add_argument("--device", type=str, nargs="+", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")

    base_args = parser.parse_args()

    for args in arg_combinations(base_args):

        print("Running with args:\n", args)

        if args.seed:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

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
            modulate_p_shuffle=args.modulate_p_shuffle,
            save_dir=args.save_dir,
            wandb_project=args.wandb_project,
            max_local_step=args.max_local_step,
            eval_iters=args.eval_iters,
            cosine_anneal=args.cosine_anneal,
            ckpt_interval=args.ckpt_interval,
            shuffle_type=args.shuffle_type,
            drift_penalty=args.drift_penalty,
            log_stats_interval=args.log_stats_interval,
            device=args.device,
            compile=args.compile,
        )

        if args.model_path:
            wm.load_model(args.model_path)

        if args.train:
            wm.train()
        elif args.generate:
            generate_text(wm.master_model)
        elif args.profile:

            cuda_is_available = torch.cuda.is_available()

            print("Warmup")
            for _ in range(5):
                wm._train_step()
                wm._shuffle_params()
                print(".", end="")

            print("Profiling shuffle_params")
            with profiler.profile(use_cuda=cuda_is_available) as prof:
                wm._shuffle_params()

            if cuda_is_available:
                torch.cuda.synchronize()

            sort_by = "cuda_time_total" if cuda_is_available else "cpu_time_total"

            print(prof.key_averages().table(sort_by=sort_by, row_limit=50))
