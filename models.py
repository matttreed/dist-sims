import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModel(nn.Module):
    def __init__(
        self, input_channels=1, input_height=28, input_width=28, num_classes=10
    ):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute the size of the input to the first fully connected layer
        # After two pooling layers, the height and width are reduced by a factor of 4 (each pooling halves the size)
        fc_input_size = 64 * (input_height // 4) * (input_width // 4)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Convolutional layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)

        # Fully connected layer 2 (output layer)
        x = self.fc2(x)

        return x


class SmallGenerativeTransformer(nn.Module):
    def __init__(
        self, vocab_size, embed_size=128, num_heads=4, num_layers=2, max_len=512
    ):
        super(SmallGenerativeTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads),
            num_layers=num_layers,
        )

        self.fc = nn.Linear(embed_size, vocab_size)

    def generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len)) == 1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x):

        tgt_mask = self.generate_causal_mask(x.size(1))

        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]

        x = x.permute(1, 0, 2)

        x = self.transformer(x, x, tgt_mask=tgt_mask)

        x = x.permute(1, 0, 2)

        x = self.fc(x)

        return x
