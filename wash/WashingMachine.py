import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import random
from tqdm import tqdm


class WashingMachine:
    def __init__(
        self,
        model_cls,
        model_kwargs,
        optimizer_cls,
        optimizer_kwargs,
        dataloader_kwargs,
        train_dataset,
        eval_dataset,
        loss_fn,
        num_workers=4,
        num_epochs=10,
        shuffle_interval=1,
        p_shuffle=0.05,
        batch_size=16,
        split_dataset=False,
        save_path="outputs/final_model.pth",
    ) -> None:
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = loss_fn
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.shuffle_interval = shuffle_interval
        self.p_shuffle = p_shuffle
        self.batch_size = batch_size
        self.split_dataset = split_dataset
        self.save_path = save_path
        self.master_model = None
        self.models = []
        self.dataloaders = []
        self.data_iters = []
        self.optimizers = []
        self.setup_master()
        self.setup_workers()

    def setup_master(self):

        self.master_model = self.model_cls(**self.model_kwargs)

    def setup_workers(self):
        if self.split_dataset:
            fracs = [
                1 / self.num_workers for _ in range(self.num_workers)
            ]  # TODO handle uneven splits
            datasets = torch.utils.data.random_split(self.train_dataset, fracs)
        else:
            datasets = [self.train_dataset for _ in range(self.num_workers)]

        for i in range(self.num_workers):
            model = self.model_cls(**self.model_kwargs)
            dataloader = DataLoader(
                datasets[i],
                batch_size=self.batch_size,
                **self.dataloader_kwargs,
            )
            optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_kwargs)
            self.models.append(model)
            self.dataloaders.append(dataloader)
            self.data_iters.append(iter(dataloader))
            self.optimizers.append(optimizer)

    def load_model(self, path):
        """
        Load a model from a given path.
        """
        self.master_model.load_state_dict(torch.load(path))
        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())

    def _shuffle_params(self):
        with torch.no_grad():
            model_params = [list(model.parameters()) for model in self.models]

            for param_idx in range(len(model_params[0])):

                size = model_params[0][param_idx].numel()

                num_to_select = int(size * self.p_shuffle)
                random_indices = torch.randperm(size)[:num_to_select]

                param_copies = [p[param_idx].clone() for p in model_params]
                rotation_amount = (
                    random.randint(1, self.num_workers - 1)
                    if self.num_workers > 1
                    else 0
                )
                param_copies = (
                    param_copies[rotation_amount:] + param_copies[:rotation_amount]
                )

                for i in range(self.num_workers):
                    model_params[i][param_idx].data.view(-1)[random_indices] = (
                        param_copies[i].view(-1)[random_indices]
                    )

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
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        print(f"Accuracy: {correct / len(self.eval_dataset)}")
        print(f"Avg Loss: {sum(cum_losses) / len(cum_losses)}")

    def _load_master_model(self):
        averaged_params = deepcopy(self.models[0].state_dict())

        # for param in averaged_params:
        #     averaged_params[param].zero_()

        # for model in self.models:
        #     model_params = model.state_dict()
        #     for param in model_params:
        #         averaged_params[param] += model_params[param] / self.num_workers

        self.master_model.load_state_dict(averaged_params)

    def _train_step(self):
        try:
            batches = [next(data_iter) for data_iter in self.data_iters]
        except StopIteration:
            self.data_iters = [iter(dl) for dl in self.dataloaders]
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

        total_iter = (
            len(self.train_dataset) / (self.num_workers * self.batch_size)
            if self.split_dataset
            else len(self.train_dataset) / (self.batch_size)
        )

        pbar = tqdm(total=total_iter)

        while True:
            try:
                avg_loss = self._train_step()
            except StopIteration:
                break

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

            if i % self.shuffle_interval == 0:
                self._shuffle_params()

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

            self._load_master_model()
            self._eval_model()

        torch.save(self.master_model.state_dict(), self.save_path)
