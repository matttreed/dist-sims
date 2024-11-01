import torch
from examples.nanogpt import GPTConfig, GPT
from transformers import AutoTokenizer

# Load the model
# gptconf = GPTConfig(block_size=256, vocab_size=50304, n_layer=2, n_head=4, n_embd=128)
gptconf = GPTConfig(block_size=512, vocab_size=50304, n_layer=12, n_head=8, n_embd=512)
model = GPT(gptconf)
model.load_state_dict(torch.load("outputs/transformer_1/ckpt_epoch_0_iter_10000.pth"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Take user input and generate text
if __name__ == "__main__":
    while True:
        user_input = input("Enter the start of your text (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break

        generated_ids = model.generate(
            torch.tensor(tokenizer.encode(user_input)).unsqueeze(0),
            max_new_tokens=500,
            temperature=0.8,
            top_k=None,
        )

        generated_text = tokenizer.decode(generated_ids.squeeze())
        print(f"Generated Text: {generated_text}")
