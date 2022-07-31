import torch
import math
import encoder
import argparse
from dyck_generator import DyckGenerator

# Copied from https://github.com/ndnlp/parity
# All arguments have a default value, but can also be given as an argument
ap = argparse.ArgumentParser()
# Length of the string given for training and testing
ap.add_argument('--train_length', dest='train_length', type=int, default=100)
ap.add_argument('--test_length', dest='test_length', type=int, default=100)
# Probabilites for the creation of the Dyck words
ap.add_argument('-p_val', dest='p_val', type=float, default=0.5, help='Probability for a new opening bracket instead of closing the upcoming bracket when generating Dyck')
ap.add_argument('-q_val', dest='q_val', type=float, default=0.5, help='Probability for not changing one character in the Dyck-word when generating it')
# Depth of the Dyck-Language (possible from 1 to 4)
ap.add_argument('--depth', dest='depth', type=int, default=1)
# number of epochs
ap.add_argument('--epochs', dest='epochs', type=int, default=100)
# number of different steps for training and testing
ap.add_argument('--steps', dest='steps', type=int, default=100)
############ Parameters for the model ############
# Number of layers and heads in the Transformer
ap.add_argument('--layers', dest='layers', type=int, default=2)
ap.add_argument('--heads', dest='heads', type=int, default=2)
# Dimension of the model
ap.add_argument('--d_model', type=int, default=16)
# Dimension of the Feedforward Neural Network
ap.add_argument('--d_ffnn', type=int, default=64)
# log-length scaled attention
ap.add_argument('--scaled', type=bool, default=False, help='log-length scaled attention')
# Value added to denominator in layer normalization
ap.add_argument('--eps', type=float, default=1e-5, help='Value added to denominator in layer normalization')
args = ap.parse_args()

log_sigmoid = torch.nn.LogSigmoid()

class PositionEncoding(torch.nn.Module):
    """
    A class to compute the position encoding of a given input

    ...

    Attributes
    ----------
    size : int
        size of the normally distributed vector
    scales : torch.nn.Parameter
        A torch parameter containing a tensor of random numbers drawn from a normal distribution with mean 0 and std 1

    Methods
    -------
    forward(n)
        Computes the positional encoding
    """

    def __init__(self, size):
        super().__init__()
        assert size % 2 == 0
        self.size = size
        self.scales = torch.nn.Parameter(torch.normal(0, 1., (size,)))

    def forward(self, n):
        # create positional encoding

        p = torch.arange(0, n).to(torch.float).unsqueeze(1)
        pe = torch.cat([
            p / n * torch.exp(self.scales[:n//2]),
            torch.cos(p*math.pi * torch.exp(self.scales[n//2:])),
        ], dim=1)
        return pe

class Model(torch.nn.Module):
    def __init__(self, alphabet_size, layers, heads, d_model, d_ffnn, scaled=False, eps=1e-5):
        super().__init__()

        # The input layer has a word embedding and positional encoding
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)
        self.pos_encoding = PositionEncoding(d_model)

        # Call the Transformer
        if scaled:
            encoder_layer = encoder.ScaledTransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        else:
            encoder_layer = encoder.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_ffnn, dropout=0.)
        encoder_layer.norm1.eps = encoder_layer.norm2.eps = eps
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output_layer = torch.nn.Linear(d_model, 1)

    def forward(self, w):
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        y = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[-1])
        return z

model = Model(2*args.depth+1, args.layers, args.heads, args.d_model, args.d_ffnn, args.scaled, args.eps)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(args.epochs):
    train_loss = 0
    train_steps = 0
    train_correct = 0
    
    for step in range(args.steps):
        n = args.train_length
        gen = DyckGenerator(args.depth, args.p_val, args.q_val)
        inp, label = gen.generate(n)
        w = torch.tensor(inp + [2*args.depth])
        output = model(w)
        if not label: output = -output
        if output > 0: train_correct += 1
        loss = -log_sigmoid(output)
        train_loss += loss.item()
        train_steps += 1
        optim.zero_grad()
        loss.backward()
        optim.step()

    test_loss = 0
    test_steps = 0
    test_correct = 0
    for step in range(args.steps):
        n = args.test_length
        gen = DyckGenerator(args.depth, args.p_val, args.q_val)
        inp, label = gen.generate(n)
        w = torch.tensor(inp + [2*args.depth])
        output = model(w)
        if not label: output = -output
        if output > 0: test_correct += 1
        loss = -log_sigmoid(output)
        test_loss += loss.item()
        test_steps += 1
        
    print(f'train_length={args.train_length} train_ce={train_loss/train_steps/math.log(2)} train_acc={train_correct/train_steps} test_ce={test_loss/test_steps/math.log(2)} test_acc={test_correct/test_steps}', flush=True)
