import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import random
import os
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import (
    euclidean_distance,
    parameter_correlation,
    mean_squared_difference,
    cosine_similarity,
    time_function,
    drift_penalty,
    get_latest_commit_and_message,
    mimic_precision,
    IndexSelector,
    RandomIndexSelector,
    PartitionedIndexSelector,
)
import numpy as np


class WashingMachine:

    def __init__(
        self,
        model_cls,
        model_kwargs,
        train_dataset,
        # eval_dataset,
        loss_fn,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.001},
        num_workers=4,
        # num_epochs=10,
        synchronize_interval=1,
        wash_interval=1,
        ckpt_interval=None,
        eval_interval=None,
        eval_iters=50,
        p_shuffle=0.01,
        shuffle_type="shuffle",
        topology_type="full",
        batch_size=16,
        modulate_p_shuffle=False,  # Modules must be defined in order of depth
        save_dir=None,
        wandb_project=None,  # WandB project name, pass None to disable logging
        wandb_config=None,
        synchronize_method="avg",
        outer_optimizer_cls=torch.optim.SGD,
        outer_optimizer_kwargs={"lr": 0.7, "nesterov": True, "momentum": 0.9},
        drift_penalty=None,
        max_local_step=None,
        cosine_anneal=False,
        log_stats_interval=10,
        async_lag=0,
        device=None,
        compile=False,
        shuffle_quantization="float32",
        shuffle_optimizer_state=False,
        indexing_type="random",
    ) -> None:
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.outer_optimizer_cls = outer_optimizer_cls
        self.outer_optimizer_kwargs = outer_optimizer_kwargs
        self.train_dataset = train_dataset
        # self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        self.num_workers = num_workers
        # self.num_epochs = num_epochs
        self.synchronize_interval = synchronize_interval
        self.wash_interval = wash_interval
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.p_shuffle = p_shuffle
        self.shuffle_type = shuffle_type
        self.topology_type = topology_type
        self.batch_size = batch_size
        self.modulate_p_shuffle = modulate_p_shuffle
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config
        self.synchronize_method = synchronize_method
        self.drift_penalty = drift_penalty
        self.local_step = 0
        self.max_local_step = max_local_step
        self.cosine_anneal = cosine_anneal
        self.log_stats_interval = log_stats_interval
        self.async_lag = async_lag
        self.compile = compile
        self.shuffle_quantization = shuffle_quantization
        self.shuffle_optimizer_state = shuffle_optimizer_state
        self.indexing_type = indexing_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not device else device

        self.master_model = None
        self.master_optimizer = None
        self.models = []
        self.dataloader = None
        self.data_iter = None
        self.optimizers = []
        self.schedulers = []
        self._setup_master()
        self._setup_workers()
        if self.indexing_type == "partitions":
            self.index_selector: IndexSelector = PartitionedIndexSelector(
                self.master_model.parameters(), p=self.p_shuffle
            )
        elif self.indexing_type == "random":
            self.index_selector: IndexSelector = RandomIndexSelector(self.master_model.parameters(), p=self.p_shuffle)

        self.async_queue = [[] for _ in range(len(list(self.models[0].parameters())))]
        # self.parameter_correlation = parameter_correlation
        # self.euclidean_distance = euclidean_distance
        # self.mean_squared_difference = mean_squared_difference
        # self.cosine_similarity = cosine_similarity

        self.losses = []
        self.grad_norms = []

        assert self.synchronize_method in ["avg", "diloco"], "Invalid synchronization method"

        assert self.topology_type in ["full", "ring"], "Invalid topology type"

        assert self.shuffle_type in ["shuffle", "avg"], "Invalid shuffle type"

        assert self.indexing_type in ["random", "partitions"], "Invalid indexing type"

        commit_hash, commit_message = get_latest_commit_and_message()

        print(f"Commit Hash: {commit_hash}")
        print(f"Commit Message: {commit_message}")

        wandb_config = {
            # "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "inner_learning_rate": self.optimizer_kwargs.get("lr", None),
            "outer_learning_rate": self.outer_optimizer_kwargs.get("lr", None),
            "synchronize_method": self.synchronize_method,
            "num_workers": self.num_workers,
            "synchronize_interval": self.synchronize_interval,
            "p_shuffle": self.p_shuffle,
            "modulate_p_shuffle": self.modulate_p_shuffle,
            "shuffle_type": self.shuffle_type,
            "drift_penalty": self.drift_penalty,
            "wash_interval": self.wash_interval,
            "cosine_anneal": self.cosine_anneal,
            "max_local_step": self.max_local_step,
            "eval_iters": self.eval_iters,
            "save_dir": self.save_dir,
            "model_kwargs": self.model_kwargs,
            "commit_hash": commit_hash,
            "commit_message": commit_message,
            "topology_type": self.topology_type,
            "async_lag": self.async_lag,
            "shuffle_quantization": self.shuffle_quantization,
            "shuffle_optimizer_state": self.shuffle_optimizer_state,
            "indexing_type": self.indexing_type,
        }

        if self.wandb_project:
            wandb.init(project=self.wandb_project, config=self.wandb_config)
            wandb.config.update(wandb_config)

    def _setup_master(self):

        self.master_model = self.model_cls(**self.model_kwargs).to(self.device)

        # Grad is set manually for diloco
        for param in self.master_model.parameters():
            param.requires_grad = True

        if self.synchronize_method == "diloco":
            self.master_optimizer = self.outer_optimizer_cls(
                self.master_model.parameters(), **self.outer_optimizer_kwargs
            )

    def _setup_workers(self):
        self.dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.data_iter = iter(self.dataloader)

        for i in range(self.num_workers):
            # model = self.model_cls(**self.model_kwargs)
            # TODO (is this best thing to do? might be necessary when only sharing top grads)
            model = deepcopy(self.master_model).to(self.device)
            optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_kwargs)
            self.models.append(model)
            self.optimizers.append(optimizer)
            if self.cosine_anneal:
                self.schedulers.append(CosineAnnealingLR(optimizer, T_max=self.max_local_step))
            else:
                self.schedulers.append(None)

    def _save_model(self):
        if not self.save_dir:
            return

        name = f"iter_{self.local_step}"
        self._load_master_model()
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.master_model.state_dict(), f"{self.save_dir}/avg_{name}.pth")
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{self.save_dir}/model_{i}_{name}.pth")

    def _shuffle_params(self):
        if self.num_workers == 1 or self.p_shuffle == 0:
            return
        with torch.no_grad():
            model_params = [list(model.parameters()) for model in self.models]

            L = len(model_params[0])

            for param_idx in range(L):
                p_shuffle = (
                    self.p_shuffle if not self.modulate_p_shuffle else self.p_shuffle * (1 - param_idx / (L - 1))
                )
                p_shuffle /= 1 - 1 / self.num_workers  # Account for poss of shuffling to self.

                size = model_params[0][param_idx].numel()
                masked_indices = torch.bernoulli(torch.full((size,), p_shuffle, device=self.device)).bool()
                num_masked = masked_indices.sum()

                if self.topology_type == "full":
                    permutation_tensor = torch.rand(num_masked, self.num_workers, device=self.device).argsort(dim=1)
                elif self.topology_type == "ring":
                    permutation_tensor = torch.arange(self.num_workers, device=self.device).repeat(num_masked, 1)
                    random_shifts = torch.randint(0, 2, (num_masked,), device=self.device) * 2 - 1
                    permutation_tensor = (permutation_tensor + random_shifts.view(-1, 1)) % self.num_workers

                row_indices = permutation_tensor.T
                column_indices = torch.arange(num_masked, device=self.device).expand(self.num_workers, -1)

                new_params = torch.stack([param[param_idx].view(-1)[masked_indices] for param in model_params])[
                    row_indices, column_indices
                ]
                new_params = mimic_precision(new_params, precision=self.shuffle_quantization)

                for model_idx in range(self.num_workers):
                    updated_param = new_params[model_idx]
                    model_params[model_idx][param_idx].view(-1).masked_scatter_(masked_indices, updated_param)

    def _avg_params(self):
        if self.num_workers == 1 or self.p_shuffle == 0:
            return
        with torch.no_grad():
            model_params = [list(model.parameters()) for model in self.models]
            master_model_params = list(self.master_model.parameters())  # used only for index selector

            L = len(model_params[0])

            for param_idx in range(L):
                p_shuffle = (
                    self.p_shuffle if not self.modulate_p_shuffle else self.p_shuffle * (1 - param_idx / (L - 1))
                )
                if self.shuffle_type == "shuffle":
                    p_shuffle /= 1 - 1 / self.num_workers  # Account for poss of shuffling to self.

                masked_indices = self.index_selector.get_indices(master_model_params[param_idx])
                num_masked = masked_indices.sum()

                if num_masked == 0:
                    continue

                if self.topology_type == "full":
                    new_params = (
                        mimic_precision(
                            torch.stack([param[param_idx][masked_indices] for param in model_params]),
                            precision=self.shuffle_quantization,
                        )
                        .mean(dim=0)
                        .unsqueeze(0)
                        .repeat(self.num_workers, 1)
                    )

                    new_exp_avg = mimic_precision(
                        torch.stack(
                            [
                                self.optimizers[model_idx].state[model_params[model_idx][param_idx]]["exp_avg"][
                                    masked_indices
                                ]
                                for model_idx in range(self.num_workers)
                            ]
                        )
                    )
                    new_exp_avg_weights = torch.tensor(
                        [
                            self.optimizers[model_idx]
                            .state[model_params[model_idx][param_idx]]["exp_avg"]
                            .view(-1)
                            .norm()
                            .item()
                            for model_idx in range(self.num_workers)
                        ],
                        device=self.device,
                    )
                    new_exp_avg_weights /= new_exp_avg_weights.sum()

                    new_exp_avg *= new_exp_avg_weights.unsqueeze(1)
                    new_exp_avg = new_exp_avg.sum(dim=0)

                    new_exp_avg_sq = mimic_precision(
                        torch.stack(
                            [
                                self.optimizers[model_idx]
                                .state[model_params[model_idx][param_idx]]["exp_avg_sq"][masked_indices]
                                .view(-1)
                                .sqrt()
                                for model_idx in range(self.num_workers)
                            ]
                        )
                    )
                    # be careful of taking var for parameters with 1 or 0 val?
                    new_exp_avg_sq_weights = torch.tensor(
                        [
                            self.optimizers[model_idx]
                            .state[model_params[model_idx][param_idx]]["exp_avg_sq"]
                            .view(-1)
                            .var()
                            .item()
                            for model_idx in range(self.num_workers)
                        ],
                        device=self.device,
                    )
                    new_exp_avg_sq_weights /= new_exp_avg_sq_weights.sum()

                    new_exp_avg_sq *= new_exp_avg_sq_weights.unsqueeze(1)
                    new_exp_avg_sq = new_exp_avg_sq.sum(dim=0).pow(2)
                elif self.topology_type == "ring":
                    new_params = torch.zeros(self.num_workers, num_masked, device=self.device)
                    new_exp_avg = torch.zeros(self.num_workers, num_masked, device=self.device)
                    new_exp_avg_sq = torch.zeros(self.num_workers, num_masked, device=self.device)

                    for model_idx in range(self.num_workers):
                        # Get indices of neighbors in the ring
                        left_neighbor_idx = (model_idx - 1) % self.num_workers  # Wrap around to the last model
                        right_neighbor_idx = (model_idx + 1) % self.num_workers  # Wrap around to the first model

                        # Stack the parameters of the current model and its neighbors
                        neighbor_params = mimic_precision(
                            torch.stack(
                                [
                                    model_params[left_neighbor_idx][param_idx][masked_indices],
                                    model_params[model_idx][param_idx][masked_indices],
                                    model_params[right_neighbor_idx][param_idx][masked_indices],
                                ]
                            ),
                            precision=self.shuffle_quantization,
                        )

                        # Compute the average for the current model
                        new_params[model_idx] = neighbor_params.mean(dim=0)

                        neighbor_exp_avg = mimic_precision(
                            torch.stack(
                                [
                                    self.optimizers[left_neighbor_idx].state[
                                        model_params[left_neighbor_idx][param_idx]
                                    ]["exp_avg"][masked_indices],
                                    self.optimizers[model_idx].state[model_params[model_idx][param_idx]]["exp_avg"][
                                        masked_indices
                                    ],
                                    self.optimizers[right_neighbor_idx].state[
                                        model_params[right_neighbor_idx][param_idx]
                                    ]["exp_avg"][masked_indices],
                                ]
                            ),
                            precision=self.shuffle_quantization,
                        )

                        new_exp_avg[model_idx] = neighbor_exp_avg.mean(dim=0)

                        neighbor_exp_avg_sq = mimic_precision(
                            torch.stack(
                                [
                                    self.optimizers[left_neighbor_idx].state[
                                        model_params[left_neighbor_idx][param_idx]
                                    ]["exp_avg_sq"][masked_indices],
                                    self.optimizers[model_idx].state[model_params[model_idx][param_idx]]["exp_avg_sq"][
                                        masked_indices
                                    ],
                                    self.optimizers[right_neighbor_idx].state[
                                        model_params[right_neighbor_idx][param_idx]
                                    ]["exp_avg_sq"][masked_indices],
                                ]
                            ),
                            precision=self.shuffle_quantization,
                        )

                        new_exp_avg_sq[model_idx] = neighbor_exp_avg_sq.mean(dim=0)

                lr = self.optimizer_kwargs.get("lr", 1)

                # for model_idx in range(self.num_workers):
                #     momentum = (
                #         self.optimizers[model_idx]
                #         .state[model_params[model_idx][param_idx]]["exp_avg"]
                #         .view(-1)[masked_indices]
                #     )
                #     new_params[model_idx] -= momentum * self.async_lag * lr * 0.5

                # Compute pseudo gradients for each model
                # pseudo_gradients = torch.stack(
                #     [
                #         (model_params[model_idx][param_idx].view(-1)[masked_indices] - new_params[model_idx])
                #         * self.p_shuffle
                #         / lr
                #         for model_idx in range(self.num_workers)
                #     ]
                # )
                new_params = mimic_precision(new_params, precision=self.shuffle_quantization)
                new_exp_avg = mimic_precision(new_exp_avg, precision=self.shuffle_quantization)
                new_exp_avg_sq = mimic_precision(new_exp_avg_sq, precision=self.shuffle_quantization)
                self.async_queue[param_idx].append((new_params, new_exp_avg, new_exp_avg_sq, masked_indices))

                if len(self.async_queue[param_idx]) > self.async_lag:
                    new_params, new_exp_avg, new_exp_avg_sq, masked_indices = self.async_queue[param_idx].pop(0)
                    for model_idx in range(self.num_workers):
                        param = model_params[model_idx][param_idx]
                        param.masked_scatter_(masked_indices, new_params)

                        if self.shuffle_optimizer_state:
                            state = self.optimizers[model_idx].state[param]
                            state["exp_avg"].masked_scatter_(masked_indices, new_exp_avg)
                            state["exp_avg_sq"].masked_scatter_(masked_indices, new_exp_avg_sq)

                    # if model_params[model_idx][param_idx].grad is not None:
                    #     model_params[model_idx][param_idx].grad.masked_scatter_(
                    #         masked_indices, pseudo_gradients[model_idx]
                    #     )

                # beta1, _ = self.optimizers[0].defaults.get("betas", None)  # TODO make work for param groups

                # # update optimizers
                # for model_idx in range(self.num_workers):
                #     state = self.optimizers[model_idx].state[model_params[model_idx][param_idx]]
                #     momentum = state["exp_avg"].view(-1)
                #     momentum[masked_indices] = momentum[masked_indices] * beta1 + pseudo_gradients[
                #         model_idx
                #     ] * self.p_shuffle * (1 - beta1)

    def _outer_step(self):

        self.master_optimizer.zero_grad()

        delta = {name: torch.zeros_like(param.data) for name, param in self.master_model.named_parameters()}
        for local_model in self.models:
            for name, param in local_model.named_parameters():
                delta[name] += param.data - self.master_model.state_dict()[name].data

        for name, param in self.master_model.named_parameters():
            delta[name] /= self.num_workers
            # param.data += delta[name]
            param.grad = -delta[name]
            # / (
            #     self.optimizer_kwargs.get("lr")
            #     * self.synchronize_interval  # scaling delta to match grad (roughly)
            # )

        self.master_optimizer.step()

        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())

    def _synchronize_models(self):
        if self.synchronize_method == "avg":
            self._load_master_model()
            for model in self.models:
                model.load_state_dict(self.master_model.state_dict())
        elif self.synchronize_method == "diloco":
            self._outer_step()

    def _load_master_model(self):
        with torch.no_grad():
            for param_name, param in self.master_model.named_parameters():
                param.zero_()
                for model in self.models:
                    param += model.state_dict()[param_name] / self.num_workers

    # def _analyze_models(self):
    #     param_correlation = parameter_correlation(self.models)
    #     euclidean_dist = euclidean_distance(self.models)
    #     print(f"Parameter Correlation: {param_correlation:.4f}")
    #     print(f"Euclidean Distance: {euclidean_dist:.4f}")

    #     if self.wandb_project:
    #         wandb.log(
    #             {
    #                 "global_step": self.local_step * self.num_workers,
    #                 "local_step": self.local_step,
    #                 "correlation": param_correlation,
    #             }
    #         )

    def _log_stats(self):
        # all_model_params = torch.cat([param.view(-1) for param in self.models[0].parameters()])
        # avg_weight_value = all_model_params.abs().mean().item()
        # std_weight_value = all_model_params.abs().std().item()
        # print(f"Avg Weight Value: {avg_weight_value:.4f}")
        # print(f"Std Weight Value: {std_weight_value:.4f}")
        if not self.wandb_project:
            return

        cum_grad_norm_var = np.var(self.grad_norms)
        sliding_grad_norm_var = np.var(self.grad_norms[-100:])
        cum_loss_var = np.var(self.losses)
        sliding_loss_var = np.var(self.losses[-100:])
        # param_correlation = parameter_correlation(self.models)
        # euclidean_dist = euclidean_distance(self.models)
        # print(f"Parameter Correlation: {param_correlation:.4f}")
        # print(f"Euclidean Distance: {euclidean_dist:.4f}")

        wandb.log(
            {
                "global_step": self.local_step * self.num_workers,
                "local_step": self.local_step,
                "lr": self.optimizers[0].param_groups[0]["lr"],
                "loss": random.choice(self.losses[-self.num_workers :]),
                "grad_norm": random.choice(self.grad_norms[-self.num_workers :]),
                "cum_grad_norm_var": cum_grad_norm_var,
                "sliding_grad_norm_var": sliding_grad_norm_var,
                "cum_loss_var": cum_loss_var,
                "sliding_loss_var": sliding_loss_var,
                "p_shuffle": self.p_shuffle,
                # "param_correlation": param_correlation,
                # "euclidean_dist": euclidean_dist,
            }
        )

    def _eval_model(self):
        self._load_master_model()
        # self.master_model = deepcopy(self.models[0])
        self.master_model.eval()
        for model in self.models:
            model.eval()
        # print(euclidean_distance([self.master_model] + self.models))
        # print(parameter_correlation([self.master_model] + self.models))
        # print(self.master_model.state_dict().keys())

        # correct = 0
        master_losses = []
        local_losses = []
        ensemble_losses = []
        with torch.no_grad():
            for i in range(self.eval_iters):  # TODO MAGIC NUMBER
                x, y = next(self.data_iter)
                x, y = x.to(self.device), y.to(self.device)

                master_output = self.master_model(x)
                master_loss = self.loss_fn(master_output, y)
                master_losses.append(master_loss.item())

                ensemble_logits = torch.zeros_like(master_output)

                for model in self.models:
                    local_output = model(x)
                    ensemble_logits += local_output / self.num_workers
                    local_loss = self.loss_fn(local_output, y)
                    local_losses.append(local_loss.item())

                ensemble_loss = self.loss_fn(ensemble_logits, y)
                ensemble_losses.append(ensemble_loss.item())

        avg_master_loss = sum(master_losses) / len(master_losses)
        master_loss_std = np.std(master_losses)
        avg_local_loss = sum(local_losses) / len(local_losses)
        local_loss_std = np.std(local_losses)
        avg_ensemble_loss = sum(ensemble_losses) / len(ensemble_losses)
        ensemble_loss_std = np.std(ensemble_losses)

        print(f"Avg Loss: {avg_master_loss:.4f}")

        if self.wandb_project:
            wandb.log(
                {
                    "global_step": self.local_step * self.num_workers,
                    "local_step": self.local_step,
                    "master_loss": avg_master_loss,
                    "master_loss_std": master_loss_std,
                    "local_loss": avg_local_loss,
                    "local_loss_std": local_loss_std,
                    "ensemble_loss": avg_ensemble_loss,
                    "ensemble_loss_std": ensemble_loss_std,
                }
            )

        for model in self.models:
            model.train()

    def _train_step(self):
        if self.max_local_step and self.local_step >= self.max_local_step:
            raise StopIteration("End of Epoch")

        try:

            for model, optimizer, scheduler in zip(self.models, self.optimizers, self.schedulers):
                x, y = next(self.data_iter)
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = self.loss_fn(output, y)
                if self.synchronize_method == "diloco" and self.drift_penalty:
                    loss += drift_penalty(model, self.master_model, self.drift_penalty)
                loss.backward()
                optimizer.step()
                if self.cosine_anneal:
                    scheduler.step()

                self.losses.append(loss.item())
                self.grad_norms.append(
                    torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad != None])).item()
                )

        except StopIteration:
            self.data_iter = iter(self.dataloader)
            raise StopIteration("End of Epoch")

    def _train_loop(self):

        total_iter = len(self.train_dataset) / (self.num_workers * self.batch_size)

        total_iter = min(total_iter, self.max_local_step)

        pbar = tqdm(total=total_iter)

        while True:

            try:
                self._train_step()
            except StopIteration:
                break

            loss = random.choice(self.losses[-self.num_workers :])

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss:.4f}"})

            if self.synchronize_interval and self.local_step % self.synchronize_interval == 0:
                self._synchronize_models()

            if self.wash_interval and self.local_step % self.wash_interval == 0:
                if self.shuffle_type == "shuffle":
                    self._shuffle_params()
                else:
                    self._avg_params()

            if self.eval_interval and self.local_step % self.eval_interval == 0:
                self._eval_model()

            if self.ckpt_interval and self.local_step % self.ckpt_interval == 0 and self.local_step > 0:
                self._save_model()

            if self.local_step % self.log_stats_interval == 0:
                self._log_stats()

            self.local_step += 1

        pbar.close()

    def train(self):
        for model in self.models:
            model.train()

        self._train_loop()

        self._eval_model()

        self._save_model()

        if self.wandb_project:
            wandb.finish()

    def load_model(self, path):
        self.master_model.load_state_dict(torch.load(path))
        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())
