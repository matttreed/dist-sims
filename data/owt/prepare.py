import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from multiprocessing import cpu_count


# Load the OpenWebText dataset from HuggingFace
def download_openwebtext():
    dataset = load_dataset("openwebtext", split="train")
    return dataset


# Tokenize the dataset and append the EOT token
def tokenize_dataset_with_eot(dataset, tokenizer, eot_token_id):
    def tokenize_function(example):
        tokenized = tokenizer(example["text"], truncation=True, padding=False)
        # Append EOT token ID to the end of each tokenized input
        tokenized["input_ids"].append(eot_token_id)
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, batched=False, num_proc=cpu_count()
    )
    return tokenized_dataset


# Save the tokenized dataset into a .bin file
def save_tokenized_to_bin(tokenized_dataset, output_file):
    with open(output_file, "wb") as f:
        for entry in tokenized_dataset:
            input_ids = entry["input_ids"]
            # Convert PyTorch Tensor to NumPy array
            input_ids_np = np.array(
                input_ids, dtype=np.int32
            )  # Ensure it's an integer type
            input_ids_np.tofile(f)  # Now we can use tofile


def main():
    # Step 1: Download the OpenWebText dataset
    dataset = download_openwebtext()

    # Step 2: Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 tokenizer

    # Get the EOT token ID
    eot_token_id = tokenizer.eos_token_id  # Typically 50256 for GPT-2

    # Step 3: Tokenize the dataset and add EOT tokens
    tokenized_dataset = tokenize_dataset_with_eot(dataset, tokenizer, eot_token_id)

    # Step 4: Save the tokenized dataset to a .bin file
    output_file = "openwebtext_tokenized_with_eot.bin"
    save_tokenized_to_bin(tokenized_dataset, output_file)
    print(f"Tokenized dataset with EOT saved to {output_file}")


if __name__ == "__main__":
    main()
