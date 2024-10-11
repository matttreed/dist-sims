import torch

import time


def extract_parameters_by_indices(model, indices_dict):
    """
    Extract parameter values from the model based on the given indices.
    """
    extracted_params = {}
    for name, param in model.named_parameters():
        param_flat = param.view(-1)
        extracted_params[name] = param_flat[indices_dict[name]].clone().detach()
    return extracted_params


def insert_parameters_by_indices(model, indices_dict, param_values):
    """
    Insert parameter values into the model at the given indices.
    """
    for name, param in model.named_parameters():
        param_flat = param.view(-1).detach().clone()
        for idx, val in zip(indices_dict[name], param_values[name]):
            param_flat[idx] = val

        with torch.no_grad():
            param.data.copy_(param_flat.view_as(param))


def train_epoch(model, optimizer, dataloader, loss_fn, shuffle_interval, conn):
    for i, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # print(f"loss: {loss.item()}")

        if i % shuffle_interval == 0:

            indices_dict = conn.recv()
            param_subset = extract_parameters_by_indices(model, indices_dict)

            conn.send(param_subset)
            updated_params = conn.recv()

            insert_parameters_by_indices(model, indices_dict, updated_params)

    conn.send(model.state_dict())


def run_trainer(
    rank,
    model_cls,
    model_kwargs,
    optimizer_cls,
    optimizer_kwargs,
    dataloader_kwargs,
    loss_fn,
    dataset,
    batch_size,
    num_epochs,
    conn,
    shuffle_interval,
    cpu_times,
    lock,
):
    """
    Worker function to simulate training and sending/receiving weights.
    """
    print(f"Worker {rank}: Starting training")

    start_cpu_time = time.process_time()

    try:

        torch.manual_seed(rank)

        model = model_cls(**model_kwargs)
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs
        )

        for epoch in range(num_epochs):
            train_epoch(model, optimizer, dataloader, loss_fn, shuffle_interval, conn)

        cpu_time_taken = time.process_time() - start_cpu_time

        with lock:
            cpu_times[rank + 1] = cpu_time_taken

    except Exception as e:
        print(f"Worker {rank} encountered an error: {e}")
    finally:
        conn.close()
        print(f"Worker {rank}: Pipe closed.")
