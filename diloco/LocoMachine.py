import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import random
from tqdm import tqdm


class LocoMachine:

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
        sync_interval=1,
        batch_size=16,
        save_path=None,
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
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self.save_path = save_path
        self.master_model = None
        self.master_optimizer = None
        self.dataloader = None
        self.data_iter = None
        self.models = []
        self.optimizers = []
        self.setup_master()
        self.setup_workers()

    def setup_master(self):

        self.master_model = self.model_cls(**self.model_kwargs)
        self.master_optimizer = self.outer_optimizer_cls(
            self.master_model.parameters(), **self.outer_optimizer_kwargs
        )

    def setup_workers(self):
        self.dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, **self.dataloader_kwargs
        )
        self.data_iter = iter(self.dataloader)

        for i in range(self.num_workers):
            model = deepcopy(self.master_model)
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
            param.grad = -delta[name]  # This line

        self.master_optimizer.step()

        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())

    def _eval_model(self):
        self.master_model.eval()

        dataloader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=self.batch_size, **self.dataloader_kwargs
        )

        correct = 0
        cum_losses = []
        with torch.no_grad():
            for data, target in dataloader:
                output = self.master_model(data)
                loss = self.loss_fn(output, target)
                cum_losses.append(loss.item())
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # TODO implement accuracy for arbitrary output type
                correct += pred.eq(target.view_as(pred)).sum().item()

        print(f"Accuracy: {correct / len(self.eval_dataset)}")
        print(f"Avg Loss: {sum(cum_losses) / len(cum_losses)}")

    def _inner_step(self):
        try:
            batches = [next(self.data_iter) for _ in range(self.num_workers)]
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            raise StopIteration("End of Epoch")

        avg_loss = 0

        for model, (x, y), optimizer in zip(self.models, batches, self.optimizers):
            optimizer.zero_grad()
            output = model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / self.num_workers

        return avg_loss

    def _train_epoch(self):
        i = 0

        pbar = tqdm(
            total=len(self.train_dataset) / (self.batch_size * self.num_workers)
        )

        while True:
            try:
                avg_loss = self._inner_step()
            except StopIteration:
                break

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

            if i % self.sync_interval == 0:
                self._outer_step()

            i += 1

        pbar.close()

    def train(self):
        """
        Train the model.
        """
        for model in self.models:
            model.train()

        for epoch in range(self.num_epochs):
            print(f"Master: Epoch {epoch+1}")
            self._train_epoch()
            self._eval_model()

        if self.save_path:
            torch.save(self.master_model.state_dict(), self.save_path)
