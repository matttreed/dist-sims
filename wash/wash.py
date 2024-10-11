import torch
import multiprocessing as mp
import signal
from wash.trainer import run_trainer
from wash.coordinator import run_coordinator


def signal_handler(signum, frame, processes):
    print(f"Received signal {signum}, terminating child processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join()
    print("All child processes terminated.")
    exit(1)


def simulate_wash(
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
):

    dataset_length = len(train_dataset)

    if split_dataset:
        dataset_length = dataset_length // num_workers
        lengths = [dataset_length] * num_workers  # TODO handle uneven splits
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
        target=run_coordinator,
        args=(
            num_workers,
            master_pipes,
            model_cls,
            model_kwargs,
            dataloader_kwargs,
            loss_fn,
            eval_dataset,
            dataset_length,
            batch_size,
            p_shuffle,
            num_epochs,
            shuffle_interval,
            cpu_times,
            lock,
            save_path,
        ),
    )
    processes.append(master)

    for i in range(num_workers):
        p = mp.Process(
            target=run_trainer,
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
        print(
            "Master process (commmunication): ", cpu_times[0]
        )  # TODO: this actually isn't really right (most of the time spent on washing is on worker processes)
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

        del cpu_times
        print("All processes cleaned up.")
