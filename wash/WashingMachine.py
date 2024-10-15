import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import random
from tqdm import tqdm


class WashingMachine:
    """
    A class for distributed training of models with parameter shuffling across workers.

    Parameters:
    ----------
    model_cls : class
        The class of the model to be instantiated for each worker (e.g., a neural network class).
    model_kwargs : dict
        A dictionary of keyword arguments to pass to the model constructor (e.g., layer sizes).
    train_dataset : torch.utils.data.Dataset
        The training dataset used by all workers.
    eval_dataset : torch.utils.data.Dataset
        The evaluation dataset used to test the models during or after training.
    loss_fn : callable
        The loss function used to compute the loss (e.g., `torch.nn.CrossEntropyLoss()`).
    optimizer_cls : class, optional
        The optimizer class (e.g., `torch.optim.AdamW`). Default is `torch.optim.AdamW`.
    optimizer_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the optimizer (e.g., learning rate). Default is {"lr": 0.001}.
    dataloader_kwargs : dict, optional
        Additional arguments to pass to the `DataLoader` (e.g., `num_workers`, `pin_memory`). Default is an empty dictionary.
    num_workers : int, optional
        The number of model workers to train in parallel. Default is 4.
    num_epochs : int, optional
        The number of training epochs. Default is 10.
    shuffle_interval : int, optional
        How often (in epochs) to shuffle parameters between the models. Default is 1.
    p_shuffle : float, optional
        The base probability of shuffling each parameter. Default is 0.01.
    batch_size : int, optional
        The batch size for training and evaluation. Default is 16.
    split_dataset : bool, optional
        Whether to split the dataset across workers, ensuring each worker trains on different data. Default is False.
    modulate_p_shuffle : bool, optional
        Whether to modulate the shuffling probability based on layer depth. Default is True.
    save_path : str, optional
        Path to save the model checkpoints. If `None`, no checkpoints are saved. Default is None.

    Attributes:
    ----------
    master_model : nn.Module
        The master model used as the central reference for creating worker models.
    models : list of nn.Module
        A list containing the models for each worker.
    dataloaders : dict
        A dictionary containing the data loaders for 'train' and 'eval' datasets.
    data_iters : list
        A list of iterators for cycling through data in parallel across workers.
    optimizers : list of optim.Optimizer
        A list of optimizers for each worker.
    """

    def __init__(
        self,
        model_cls,
        model_kwargs,
        train_dataset,
        eval_dataset,
        loss_fn,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.001},
        dataloader_kwargs={},
        num_workers=4,
        num_epochs=10,
        shuffle_interval=1,
        p_shuffle=0.01,
        batch_size=16,
        split_dataset=False,
        modulate_p_shuffle=True,  # Modules must be defined in order of depth
        save_path=None,
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
        self.modulate_p_shuffle = modulate_p_shuffle
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

    def _load_master_model(self):
        # TODO add different averaging methods
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

        if self.save_path:
            torch.save(self.master_model.state_dict(), self.save_path)
