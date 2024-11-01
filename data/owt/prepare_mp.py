import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Import tqdm for progress tracking


# Load the OpenWebText dataset from Hugging Face
def download_openwebtext():
    dataset = load_dataset("openwebtext", split="train")
    return dataset


# Tokenize a single example with the EOT token appended
def tokenize_example(example, eot_token_id, tokenizer):
    tokenized = tokenizer(example["text"], truncation=True, padding=False)
    # Append EOT token ID to the end of each tokenized input
    tokenized["input_ids"].append(eot_token_id)
    return tokenized["input_ids"]


# Process a subset of the dataset in parallel
def process_chunk(start_idx, end_idx, dataset, eot_token_id, tokenizer):
    tokenized_chunk = []
    for i in range(start_idx, end_idx):
        tokenized_chunk.append(tokenize_example(dataset[i], eot_token_id, tokenizer))
    return tokenized_chunk


# Write all tokenized data to a single binary file with progress
def save_to_bin_file(tokenized_data, output_file):
    with open(output_file, "ab") as f:  # 'ab' mode to append
        for input_ids in tokenized_data:
            input_ids_np = np.array(input_ids, dtype=np.int32)
            input_ids_np.tofile(f)


def main():
    # Step 1: Download the OpenWebText dataset
    dataset = download_openwebtext()

    # Step 2: Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eot_token_id = tokenizer.eos_token_id  # Typically 50256 for GPT-2

    # Step 3: Split dataset into chunks for parallel processing
    num_cpus = cpu_count()
    chunk_size = len(dataset) // num_cpus
    print(f"Using {num_cpus} CPUs for tokenization.")

    # Step 4: Tokenize in parallel with progress tracking
    output_file = "openwebtext_tokenized_with_eot.bin"
    tokenized_chunks = []

    # Using tqdm to track each chunk
    with Pool(processes=num_cpus) as pool:
        # Wrap the pool.starmap call with tqdm for tracking
        for tokenized_chunk in tqdm(
            pool.starmap(
                process_chunk,
                [
                    (
                        i,
                        min(i + chunk_size, len(dataset)),
                        dataset,
                        eot_token_id,
                        tokenizer,
                    )
                    for i in range(0, len(dataset), chunk_size)
                ],
            ),
            total=num_cpus,
            desc="Tokenizing",
        ):
            tokenized_chunks.append(tokenized_chunk)

    # Step 5: Sequentially write each chunk to the binary file with progress tracking
    with tqdm(total=len(tokenized_chunks), desc="Saving to .bin file") as pbar:
        for tokenized_chunk in tokenized_chunks:
            save_to_bin_file(tokenized_chunk, output_file)
            pbar.update(1)

    print(f"Tokenized dataset with EOT saved to {output_file}")


if __name__ == "__main__":
    main()
