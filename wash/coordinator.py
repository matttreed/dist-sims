import torch
import time
import random
import math
from tqdm import tqdm


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

    weights = weights[0:1]  # TODO : make averaging configurable

    for name, param in model.named_parameters():
        avg_param = torch.zeros_like(param.data)
        for worker_weights in weights:
            avg_param += worker_weights[name]

        avg_param /= len(weights)
        param.data.copy_(avg_param)


def dispatch_indices(indices_dict, master_pipes):
    for conn in master_pipes:
        conn.send(indices_dict)


def receive_param_subsets(param_names, master_pipes):
    param_subsets = [conn.recv() for conn in master_pipes]
    return {
        name: [param[name] for param in param_subsets] for name in param_names
    }  # {param_name: [[]param values]trainers}


def shuffle_params(param_subsets):
    for key, params in param_subsets.items():
        rotation_amount = random.randint(1, len(params) - 1) if len(params) > 1 else 0
        param_subsets[key] = params[rotation_amount:] + params[:rotation_amount]


def dispatch_washed_params(param_subsets, master_pipes):
    for i, conn in enumerate(master_pipes):
        conn.send({name: param_subsets[name][i] for name in param_subsets.keys()})


def train_epoch(
    model,
    master_pipes,
    dataset_len,
    shuffle_interval,
    batch_size,
    p_shuffle,
    num_workers,
    param_names,
):
    num_iters = math.ceil(dataset_len / batch_size)

    for i in tqdm([j for j in range(num_iters) if j % shuffle_interval == 0]):

        indices_dict = generate_random_indices(model, fraction=p_shuffle)

        dispatch_indices(indices_dict, master_pipes)
        param_subsets = receive_param_subsets(param_names, master_pipes)
        shuffle_params(param_subsets)
        dispatch_washed_params(param_subsets, master_pipes)


def run_coordinator(
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
    save_path,
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

        param_names = [name for name, param in model.named_parameters()]

        for epoch in range(num_epochs):
            print(f"Master: Epoch {epoch+1}")
            train_epoch(
                model,
                master_pipes,
                dataset_len,
                shuffle_interval,
                batch_size,
                p_shuffle,
                num_workers,
                param_names,
            )

            cpu_time_taken += time.process_time() - start_cpu_time
            start_cpu_time = time.process_time()

            load_avg_model(model, master_pipes)
            eval_model(model, eval_dataset, loss_fn, batch_size)

        with lock:
            cpu_times[0] = cpu_time_taken

        torch.save(model, save_path)

    except Exception as e:
        print(f"Master process encountered an error: {e}")
    finally:
        # Close all master pipes at the end of the process
        for pipe in master_pipes:
            pipe.close()
        print("Master process: All pipes closed.")
