import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from examples.models import SmallGenerativeTransformer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from transformer import yield_tokens

# Constants
PAD_IDX = 0
MAX_GENERATE_LEN = 50  # Max number of tokens to generate

# Load tokenizer and vocab
tokenizer = get_tokenizer("basic_english")

train_iter, _ = AG_NEWS(split=("train", "test"))
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer), specials=["<pad>", "<unk>"]
)
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)

# Load the model
model = SmallGenerativeTransformer(
    vocab_size=vocab_size, embed_size=128, num_heads=4, num_layers=2
)
model.load_state_dict(torch.load("outputs/transformer.pth"))
model.eval()


# Function to generate text
def generate_text(model, input_text, tokenizer, vocab, max_len=MAX_GENERATE_LEN):
    tokens = vocab(tokenizer(input_text))
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    for _ in range(max_len):
        with torch.no_grad():
            # Forward pass through the model to get logits
            output = model(tokens)
            next_token_logits = output[:, -1, :]  # Get logits for the last token only

            # Apply softmax to get probabilities and sample the next token
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Append the next token to the sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if the generated token is <pad> (or any other stop token)
            if next_token.item() == vocab["<pad>"]:
                break

    # Convert tokens back to text
    generated_tokens = tokens[0].tolist()  # Remove batch dimension
    generated_text = " ".join(vocab.lookup_tokens(generated_tokens))
    return generated_text


# Take user input and generate text
if __name__ == "__main__":
    while True:
        user_input = input("Enter the start of your text (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break

        generated_text = generate_text(model, user_input, tokenizer, vocab)
        print(f"Generated Text: {generated_text}")
