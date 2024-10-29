import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import random
import os
from tqdm import tqdm
import wandb
from distsims.util import (
    euclidean_distance,
    parameter_correlation,
    mean_squared_difference,
    cosine_similarity,
)


class WashingMachine:

    def __init__(
        self,
        model_cls,
        model_kwargs,
        train_dataset,
        eval_dataset,
        loss_fn,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.001},
        outer_optimizer_cls=torch.optim.SGD,
        outer_optimizer_kwargs={"lr": 0.01, "nesterov": True, "momentum": 0.9},
        dataloader_kwargs={},
        num_workers=4,
        num_epochs=10,
        synchronize_interval=1,
        ckpt_interval=None,
        eval_interval=None,
        p_shuffle=0.01,
        batch_size=16,
        data_parallel=False,
        modulate_p_shuffle=True,  # Modules must be defined in order of depth
        save_dir=None,
        wandb_project=None,  # WandB project name, pass None to disable logging
        wandb_config=None,
        synchronize_method="wash",
        drift_penalty=None,
        max_local_step=None,
    ) -> None:
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.outer_optimizer_cls = outer_optimizer_cls
        self.outer_optimizer_kwargs = outer_optimizer_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.synchronize_interval = synchronize_interval
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.p_shuffle = p_shuffle
        self.batch_size = batch_size
        self.data_parallel = data_parallel
        self.modulate_p_shuffle = modulate_p_shuffle
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config
        self.synchronize_method = synchronize_method
        self.drift_penalty = drift_penalty
        self.local_step = 0
        self.max_local_step = max_local_step

        self.master_model = None
        self.master_optimizer = None
        self.models = []
        self.dataloader = None
        self.data_iter = None
        self.optimizers = []
        self._setup_master()
        self._setup_workers()

        assert self.synchronize_method in [
            "wash",
            "diloco",
        ], "Invalid synchronization method"

        assert (
            self.synchronize_method != "diloco" or self.data_parallel
        ), "Data parallelism is required for diloco"

        if self.wandb_project:
            wandb.init(project=self.wandb_project, config=self.wandb_config)
            wandb.config.update(
                {
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "inner_learning_rate": self.optimizer_kwargs.get("lr", None),
                    "outer_learning_rate": self.outer_optimizer_kwargs.get("lr", None),
                    "synchronize_method": self.synchronize_method,
                    "num_workers": self.num_workers,
                    "synchronize_interval": self.synchronize_interval,
                    "p_shuffle": self.p_shuffle,
                    "modulate_p_shuffle": self.modulate_p_shuffle,
                    "data_parallel": self.data_parallel,
                    "drift_penalty": self.drift_penalty,
                    # "block_size": self.config.block_size,
                    # "vocab_size": self.config.vocab_size,
                    # "n_layer": self.config.n_layer,
                    # "n_head": self.config.n_head,
                    # "n_embd": self.config.n_embd,
                }
            )

    def _setup_master(self):

        self.master_model = self.model_cls(**self.model_kwargs)

        # Grad is set manually for diloco
        for param in self.master_model.parameters():
            param.requires_grad = True

        if self.synchronize_method == "diloco":
            self.master_optimizer = self.outer_optimizer_cls(
                self.master_model.parameters(), **self.outer_optimizer_kwargs
            )

    def _setup_workers(self):
        self.dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, **self.dataloader_kwargs
        )
        self.data_iter = iter(self.dataloader)

        for i in range(self.num_workers):
            # model = self.model_cls(**self.model_kwargs)
            model = deepcopy(
                self.master_model
            )  # TODO (is this best thing to do? might be necessary when only sharing top grads)
            optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_kwargs)
            self.models.append(model)
            self.optimizers.append(optimizer)

    def load_model(self, path):
        """
        Load a model from a given path.
        """
        self.master_model.load_state_dict(torch.load(path))
        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())

    def _save_model(self, name):
        if not self.save_dir:
            return
        self._load_master_model()
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.master_model.state_dict(), f"{self.save_dir}/avg_{name}.pth")
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{self.save_dir}/model_{i}_{name}.pth")

    def _shuffle_params(self):
        with torch.no_grad():
            model_params = [list(model.parameters()) for model in self.models]

            L = len(model_params[0])

            for param_idx in range(L):

                p_shuffle = (
                    self.p_shuffle
                    if not self.modulate_p_shuffle
                    else self.p_shuffle * (1 - param_idx / (L - 1))
                )

                params = torch.stack(
                    [param[param_idx].view(-1) for param in model_params]
                )
                size = params.shape[1]
                permutation_tensor = torch.rand(size, self.num_workers).argsort(dim=1)

                row_indices = permutation_tensor.T
                column_indices = (
                    torch.arange(params.shape[1])
                    .unsqueeze(0)
                    .expand(params.shape[0], -1)
                )

                masked_indices = torch.nonzero(
                    torch.rand(size) < p_shuffle, as_tuple=True
                )[0]

                params[:, masked_indices] = params[row_indices, column_indices][
                    :, masked_indices
                ]

                for model_idx, updated_param in enumerate(params):
                    model_params[model_idx][param_idx].data.copy_(
                        updated_param.view_as(model_params[model_idx][param_idx])
                    )

    # according to gradient magnitude
    # def _shuffle_params(self):
    #     with torch.no_grad():
    #         model_params = [list(model.parameters()) for model in self.models]

    #         L = len(model_params[0])

    #         for param_idx in range(L):

    #             p_shuffle = (
    #                 self.p_shuffle
    #                 if not self.modulate_p_shuffle
    #                 else self.p_shuffle * (1 - param_idx / (L - 1))
    #             )

    #             params = torch.stack(
    #                 [model[param_idx].view(-1) for model in model_params]
    #             )
    #             size = params.shape[1]

    #             gradient_magnitudes = torch.stack(
    #                 [model[param_idx].grad.view(-1).abs() for model in model_params]
    #             ).sum(dim=0)

    #             masked_indices = torch.topk(gradient_magnitudes, int(p_shuffle * size))[
    #                 1
    #             ]
    #             permutation_tensor = torch.rand(size, self.num_workers).argsort(
    #                 dim=1
    #             )  # TODO: shuffle p is actually p (1-1/Num workers) since might share with self
    #             row_indices = permutation_tensor.T
    #             column_indices = (
    #                 torch.arange(params.shape[1])
    #                 .unsqueeze(0)
    #                 .expand(params.shape[0], -1)
    #             )

    #             # masked_indices = torch.nonzero(
    #             #     torch.rand(size) < p_shuffle, as_tuple=True
    #             # )[0]

    #             params[:, masked_indices] = params[row_indices, column_indices][
    #                 :, masked_indices
    #             ]

    #             for model_idx, updated_param in enumerate(params):
    #                 model_params[model_idx][param_idx].data.copy_(
    #                     updated_param.view_as(model_params[model_idx][param_idx])
    #                 )

    def _outer_step(self):

        self.master_optimizer.zero_grad()

        delta = {
            name: torch.zeros_like(param.data)
            for name, param in self.master_model.named_parameters()
        }
        for local_model in self.models:
            for name, param in local_model.named_parameters():
                delta[name] += param.data - self.master_model.state_dict()[name].data

        for name, param in self.master_model.named_parameters():
            delta[name] /= self.num_workers
            # param.data += delta[name]
            param.grad = -delta[name] / (
                self.optimizer_kwargs.get("lr")
                * self.synchronize_interval  # scaling delta to match grad (roughly)
            )

        self.master_optimizer.step()

        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())

    def _synchronize_models(self):
        if self.synchronize_method == "wash":
            self._shuffle_params()
        elif self.synchronize_method == "diloco":
            self._outer_step()

    def _load_master_model(self):
        # TODO add different averaging methods
        if self.synchronize_method == "diloco":
            return

        with torch.no_grad():
            for param_name, param in self.master_model.named_parameters():
                param.zero_()
                for model in self.models:
                    param += model.state_dict()[param_name] / self.num_workers

    # def _eval_model(self):
    #     self._load_master_model()
    #     self.master_model.eval()

    #     dataloader = torch.utils.data.DataLoader(
    #         self.eval_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #     )

    #     # correct = 0
    #     cum_losses = []
    #     with torch.no_grad():
    #         for i, (data, target) in enumerate(dataloader):
    #             output = self.master_model(data)
    #             loss = self.loss_fn(output, target)
    #             cum_losses.append(loss.item())
    #             # pred = output.argmax(
    #             #     dim=1, keepdim=True
    #             # )  # TODO implement accuracy for arbitrary output type
    #             # correct += pred.eq(target.view_as(pred)).sum().item()

    #             if i == 50:
    #                 break

    #     # accuracy = correct / len(self.eval_dataset)
    #     avg_loss = sum(cum_losses) / len(cum_losses)

    #     print(cum_losses)

    #     print(euclidean_distance([self.master_model] + self.models))
    #     print(parameter_correlation([self.master_model] + self.models))

    #     # print(f"Accuracy: {accuracy}")
    #     print(f"Avg Loss: {avg_loss}")

    #     if self.wandb_project:
    #         wandb.log({"iteration": self.local_step, "eval_loss": avg_loss})

    def _eval_model(self):
        self._load_master_model()
        # self.master_model = deepcopy(self.models[0])
        self.master_model.eval()
        # print(euclidean_distance([self.master_model] + self.models))
        # print(parameter_correlation([self.master_model] + self.models))
        # print(self.master_model.state_dict().keys())

        # correct = 0
        cum_losses = []
        with torch.no_grad():
            for i in range(50):  # TODO MAGIC NUMBER
                data, target = next(iter(self.dataloader))
                output = self.master_model(data)
                loss = self.loss_fn(output, target)
                cum_losses.append(loss.item())
                # pred = output.argmax(
                #     dim=1, keepdim=True
                # )  # TODO implement accuracy for arbitrary output type
                # correct += pred.eq(target.view_as(pred)).sum().item()

        # accuracy = correct / len(self.eval_dataset)
        avg_loss = sum(cum_losses) / len(cum_losses)

        # print(cum_losses)

        # print(f"Accuracy: {accuracy}")
        print(f"Avg Loss: {avg_loss}")

        if self.wandb_project:
            wandb.log(
                {
                    "global_step": self.local_step * self.num_workers,
                    "local_step": self.local_step,
                    "eval_loss": avg_loss,
                }
            )

    def _drift_penalty(self, model, weight=0.01):
        penalty = 0.0
        for (name, param), (_, ref_param) in zip(
            model.named_parameters(), self.master_model.named_parameters()
        ):
            # Compute the L2 norm difference with the reference parameter
            penalty += torch.norm(param - ref_param) ** 2
        return weight * penalty

    def _train_step(self):
        if self.max_local_step and self.local_step >= self.max_local_step:
            raise StopIteration("End of Epoch")

        try:
            if self.data_parallel:
                batches = [
                    next(self.data_iter) for _ in range(self.num_workers)
                ]  # Each worker gets a different batch
            else:
                batches = [
                    next(self.data_iter)
                ] * self.num_workers  # All workers get the same batch
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            raise StopIteration("End of Epoch")

        losses = []

        for model, (x, y), optimizer in zip(self.models, batches, self.optimizers):
            optimizer.zero_grad()
            output = model(x)
            loss = self.loss_fn(output, y)
            if self.synchronize_method == "diloco" and self.drift_penalty:
                loss += self._drift_penalty(model, self.drift_penalty)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return losses

    def _train_epoch(self, epoch):

        total_iter = (
            len(self.train_dataset) / (self.num_workers * self.batch_size)
            if self.data_parallel
            else len(self.train_dataset) / (self.batch_size)
        )

        total_iter = min(total_iter, self.max_local_step)

        pbar = tqdm(total=total_iter)

        while True:

            try:
                losses = self._train_step()
            except StopIteration:
                break

            if (
                self.synchronize_interval
                and self.local_step % self.synchronize_interval == 0
            ):
                self._synchronize_models()

            if (
                self.eval_interval
                and self.eval_dataset
                and self.local_step % self.eval_interval == 0
            ):
                self._eval_model()

            if (
                self.ckpt_interval
                and self.local_step % self.ckpt_interval == 0
                and self.local_step > 0
            ):
                self._save_model(f"epoch_{epoch}_iter_{self.local_step}")

            avg_loss = sum(losses) / len(losses)

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

            if self.wandb_project and self.local_step % 10 == 0:
                wandb.log(
                    {
                        "loss": avg_loss,
                        "global_step": self.local_step * self.num_workers,
                        "local_step": self.local_step,
                        # "euclidean_distance": euclidean_distance(self.models),
                        # "parameter_correlation": parameter_correlation(self.models),
                        # "mean_squared_difference": mean_squared_difference(self.models),
                        # "cosine_similarity": cosine_similarity(self.models),
                    }
                )

            self.local_step += 1

        pbar.close()

    def train(self):
        """
        Train the model.
        """
        for model in self.models:
            model.train()

        for epoch in range(self.num_epochs):
            print(f"Master: Epoch {epoch+1}")
            self._train_epoch(epoch)

            if self.eval_dataset:
                self._eval_model()

        self._save_model("final")

        if self.wandb_project:
            wandb.finish()
