import torch
import torch.nn.functional as F
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import SmallGenerativeTransformer
from wash import wash_algorithm


import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0


def CELoss(inputs, targets):
    return F.cross_entropy(
        inputs.view(-1, inputs.size(-1)), targets.view(-1), ignore_index=PAD_IDX
    )


class NextTokenDataset(Dataset):
    def __init__(self, data_iter, tokenizer, vocab, max_len=512):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data = []
        for _, text in data_iter:
            tokens = vocab(tokenizer(text))
            tokens = torch.tensor(tokens, dtype=torch.long)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]

            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            self.data.append((input_tokens, target_tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, target_tokens = self.data[idx]

        return input_tokens, target_tokens


def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=PAD_IDX)
    target_batch = pad_sequence(target_batch, batch_first=True, padding_value=PAD_IDX)
    return input_batch, target_batch


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


if __name__ == "__main__":

    train_iter, test_iter = AG_NEWS(split=("train", "test"))

    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer), specials=["<pad>", "<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    vocab_size = len(vocab)
    train_dataset = NextTokenDataset(train_iter, tokenizer, vocab, max_len=512)
    test_dataset = NextTokenDataset(test_iter, tokenizer, vocab, max_len=512)

    wash_algorithm(
        model_cls=SmallGenerativeTransformer,
        model_kwargs={
            "vocab_size": vocab_size,
            "embed_size": 128,
            "num_heads": 4,
            "num_layers": 2,
        },
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.01},
        dataloader_kwargs={"collate_fn": collate_fn},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=CELoss,
        num_workers=2,
        num_epochs=1,
        shuffle_interval=1,
        p_shuffle=0.01,
        batch_size=32,
        split_dataset=True,
    )
