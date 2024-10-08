import torch
import random
import multiprocessing as mp
import signal

import time
import math


def signal_handler(signum, frame, processes):
    print(f"Received signal {signum}, terminating child processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join()
    print("All child processes terminated.")
    exit(1)


def generate_random_indices(model, fraction=0.05):
    """
    Generate random indices for each parameter tensor in the model.
    """
    indices_dict = {}
    for name, param in model.named_parameters():
        total_elements = param.numel()
        num_indices = int(total_elements * fraction)
        random_indices = random.sample(range(total_elements), num_indices)
        indices_dict[name] = random_indices
    return indices_dict


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


def eval_model(model, eval_dataset, loss_fn, batch_size):
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)
    model.eval()

    correct = 0
    cum_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = loss_fn(output, target)
            cum_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Accuracy: {correct / len(eval_dataset)}")
    print(f"Avg Loss: {cum_loss / len(eval_dataset)}")


def load_avg_model(model, master_pipes):
    weights = []
    for conn in master_pipes:
        weights.append(conn.recv())

    for name, param in model.named_parameters():
        avg_param = torch.zeros_like(param.data)
        for worker_weights in weights:
            avg_param += worker_weights[name]

        avg_param /= len(weights)
        param.data.copy_(avg_param)


def train_worker(
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

        cpu_time_taken = time.process_time() - start_cpu_time

        with lock:
            cpu_times[rank + 1] = cpu_time_taken

    except Exception as e:
        print(f"Worker {rank} encountered an error: {e}")
    finally:
        conn.close()
        print(f"Worker {rank}: Pipe closed.")


def master_process(
    num_workers,
    master_pipes,
    model_cls,
    model_kwargs,
    loss_fn,
    eval_dataset,
    dataset_len,
    batch_size,
    p_shuffle,
    num_epochs,
    shuffle_interval,
    cpu_times,
    lock,
):
    """
    Master process that collects weights from workers, shuffles individual float values, and sends them back.
    """
    start_cpu_time = time.process_time()
    cpu_time_taken = 0

    try:

        torch.manual_seed(12345)
        random.seed(12345)
        model = model_cls(**model_kwargs)

        keys = [name for name, param in model.named_parameters()]

        for epoch in range(num_epochs):
            print(f"Master: Epoch {epoch+1}")
            for i in range(math.ceil(dataset_len / batch_size)):
                if i % shuffle_interval == 0:

                    indices_dict = generate_random_indices(model, fraction=p_shuffle)

                    for conn in master_pipes:
                        conn.send(indices_dict)
                    param_subsets = []
                    for conn in master_pipes:
                        param_subsets.append(conn.recv())

                    shuffled_params = [{} for _ in range(num_workers)]

                    for key in keys:
                        values = [param[key] for param in param_subsets]
                        rotation_amount = (
                            random.randint(1, len(values) - 1) if num_workers > 1 else 0
                        )
                        values = values[rotation_amount:] + values[:rotation_amount]
                        for i in range(len(values)):
                            shuffled_params[i][key] = values[i]

                    for i, conn in enumerate(master_pipes):
                        conn.send(shuffled_params[i])

            cpu_time_taken += time.process_time() - start_cpu_time
            start_cpu_time = time.process_time()

            load_avg_model(model, master_pipes)
            eval_model(model, eval_dataset, loss_fn, batch_size)

        with lock:
            cpu_times[0] = cpu_time_taken

        torch.save(model, "final_model.pth")

    except Exception as e:
        print(f"Master process encountered an error: {e}")
    finally:
        # Close all master pipes at the end of the process
        for pipe in master_pipes:
            pipe.close()
        print("Master process: All pipes closed.")


def wash_algorithm(
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
):

    dataset_length = len(train_dataset)

    if split_dataset:
        dataset_length = dataset_length // num_workers
        lengths = [dataset_length] * num_workers
        datasets = torch.utils.data.random_split(train_dataset, lengths)
    else:
        datasets = [train_dataset for _ in range(num_workers)]

    cpu_times = mp.Array("d", num_workers + 1)

    lock = mp.Lock()

    worker_pipes = []
    master_pipes = []
    processes = []
    for _ in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        master_pipes.append(parent_conn)
        worker_pipes.append(child_conn)

    master = mp.Process(
        target=master_process,
        args=(
            num_workers,
            master_pipes,
            model_cls,
            model_kwargs,
            loss_fn,
            eval_dataset,
            dataset_length,
            batch_size,
            p_shuffle,
            num_epochs,
            shuffle_interval,
            cpu_times,
            lock,
        ),
    )
    processes.append(master)

    for i in range(num_workers):
        p = mp.Process(
            target=train_worker,
            args=(
                i,
                model_cls,
                model_kwargs,
                optimizer_cls,
                optimizer_kwargs,
                dataloader_kwargs,
                loss_fn,
                datasets[i],
                batch_size,
                num_epochs,
                worker_pipes[i],
                shuffle_interval,
                cpu_times,
                lock,
            ),
        )
        processes.append(p)

    signal.signal(
        signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, processes)
    )
    signal.signal(
        signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame, processes)
    )

    try:
        master.start()
        for p in processes[1:]:
            p.start()

        master.join()
        for p in processes[1:]:
            p.join()

        print("All processes completed.")
        print("CPU times:")
        print("Master process (commmunication): ", cpu_times[0])
        for i in range(num_workers):
            print(f"Worker {i}: {cpu_times[i+1]}")
        print("Total time:", sum(cpu_times))

    except Exception as e:
        print(f"Main process encountered an error: {e}")
    finally:
        print("Cleaning up processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()
        print("All processes cleaned up.")
