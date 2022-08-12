import torch
import math
import random
import argparse
from dyck_generator import DyckGenerator

# All arguments have a default value, but can also be given as an argument
ap = argparse.ArgumentParser()
# Length of the string given for training and testing
ap.add_argument('--train_length', dest='train_length', type=int, default=100)
ap.add_argument('--test_length', dest='test_length', type=int, default=100)
# Probabilites for the creation of the Dyck words
ap.add_argument('--p_val', dest='p_val', type=float, default=0.5, help='Probability for a new opening bracket instead of closing the upcoming bracket when generating Dyck')
ap.add_argument('--q_val', dest='q_val', type=float, default=0.5, help='Probability for not changing one character in the Dyck-word when generating it')
# Number of different bracket types of the Dyck-Language (possible from 1 to 4)
ap.add_argument('--num_par', dest='num_par', type=int, default=1, help='the number of pairs N of N-Dyck')
# Depth for when the language Dyck-(N, D) is created
ap.add_argument('--depth', dest='depth', type=int, default=3, help='the depth D of Dyck-(N, D). Default: 3')
# If num_par bigger than 1, do we want basic N-Dyck or shuffle Dyck
ap.add_argument('--shuffle', dest='shuffle', type=bool, default=False, help='If True, than Shuffle-Dyck is produced')
# number of epochs
ap.add_argument('--epochs', dest='epochs', type=int, default=100)
# number of different steps for training and testing
ap.add_argument('--steps', dest='steps', type=int, default=100)
############ Parameters for the model ############

ap.add_argument('--big', dest='big', type=float, default=1.)
ap.add_argument('--perturb', dest='perturb', type=float, default=0, help='randomly perturb parameters')
ap.add_argument('--train', dest='train', action='store_true', default=False)
# If set to true, use hard attention instead of soft attention
ap.add_argument('--hard', type=bool, default=False, help='hard attention')
args = ap.parse_args()

log_sigmoid = torch.nn.LogSigmoid()

def _generate_mask(s):
    mask = (torch.triu(torch.ones(s, s)) == 1).transpose(0, 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionEncoding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, n, d):
        zero = torch.zeros(n)
        indicator = torch.zeros(n)
        indicator[n-1] = 1.
        pos = torch.arange(1,n+1).to(torch.float)#.unsqueeze(1)
        pe = torch.stack([zero]*3 +
                         [indicator] +
                         [d / pos] +
                         [zero]*7,
                         dim=1)
        #pe[0][4] = 0 # to fix d / 0 = inf
        return pe

class FirstLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        self.src_mask = None
        super().__init__(12, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # First head attends to all symbols,
            # second head does nothing.
            # W^Q
            [[0]*12]*12 +
            # W^K
            [[0]*12]*12 +
            # W^V
            [[ 0,0,1,0,0,0,0,0,0,0,0,0],     # count CLS (1)
             [-1,1,0,0,0,0,0,0,0,0,0,0]] +  # count 1s - 0s  (k)
            [[0]*12]*10,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(36))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*12]*5 +
            [[1,0,0,0,0,0,0,0,0,0,0,0],   # put new values into dims 5-6
             [0,1,0,0,0,0,0,0,0,0,0,0]] +
            [[0]*12]*5,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(12))

        self.linear1.weight = torch.nn.Parameter(torch.tensor([
            [0,0,0,0, 0, 1, 1,0,0,0,0,0],  # k-i-1
            [0,0,0,0, 0, 0, 1,0,0,0,0,0],  # k-i
            [0,0,0,0, 0,-1, 1,0,0,0,0,0],  # k-i+1
            [0,0,0,0,-1, 0,-1,0,0,0,0,0],  # k-d
            [0,0,0,0,-1,-1,-1,0,0,0,0,0],  # k-d-1
        ], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(5))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0, 0, 0, 0, 0]]*7 +
            [[1,-2, 1, 0, 0],  
             [0, 1,-1, 0, 0],
             [0, 0, 0, 1,-1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]], 
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(12))
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        self.src_mask = _generate_mask(src.shape[0])
        src2 = self.self_attn(src, src, src, attn_mask=self.src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class SecondLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        self.src_mask = None
        super().__init__(12, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # W^Q
            [[0]*12]*12 +
            # W^K
            [[0]*12]*12 +
            # W^V
            [[0]*12]*12,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(36))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*12]*12,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(12))

        self.linear1.weight = torch.nn.Parameter(torch.tensor([
            [-1,-1,-1,1,0,0,0,1,0,0,0,0]
        ], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(1))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0]]*10 +
            [[1]] + 
            [[0]], 
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(12))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        self.src_mask = _generate_mask(src.shape[0])
        q = src
        v = src
        src2 = self.self_attn(q, src, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class MyTransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            FirstLayer(),
            SecondLayer(),
        ])
        self.num_layers = len(self.layers)
        self.norm = None

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding = torch.eye(3, 12)
        self.pos_encoding = PositionEncoding()
        self.encoder = MyTransformerEncoder()
        self.output_layer = torch.nn.Linear(12, 1)
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,0,0,-2,-2,1,0]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w, d):
        x = self.word_embedding[w] + self.pos_encoding(len(w), d)
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = sum(self.output_layer(y))
        return z

model = Model()
optim = torch.optim.Adam(model.parameters(), lr=3e-4)

# Perturb parameters
if args.perturb > 0:
    with torch.no_grad():
        for p in model.parameters():
            p += torch.randn(p.size()) * args.perturb

if not args.train: args.epochs = 1            
for epoch in range(args.epochs):
    if args.train:
        train_loss = 0
        train_steps = 0
        train_correct = 0
        for step in range(args.steps):
            size = args.train_length
            n = args.num_par
            d = args.depth
            gen = DyckGenerator(n, args.p_val, args.q_val)
            if args.shuffle:
                inp, label = gen.generate_shuffle_dyck(size)
            else:
                inp, label = gen.generate_dyck(size, d)
            w = torch.tensor([2*n] + inp) 
            output = model(w, d)
            if not label: output = -output
            if output >= 0: train_correct += 1
            loss = -log_sigmoid(output)
            train_loss += loss.item()
            train_steps += 1
            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        test_loss = 0
        test_steps = 0
        test_correct = 0
        for step in range(args.steps):
            size = args.test_length
            n = args.num_par
            d = args.depth
            gen = DyckGenerator(n, args.p_val, args.q_val)
            if args.shuffle:
                inp, label = gen.generate_shuffle_dyck(size)
            else:
                inp, label = gen.generate_dyck(size, d)
            w = torch.tensor([2*n] + inp)
            output = model(w, d)
            if not label: output = -output
            if output >= 0: test_correct += 1
            loss = -log_sigmoid(output)
            test_loss += loss.item()
            test_steps += 1

    if args.train:
        print(f'train_length={args.train_length} train_ce={train_loss/train_steps/math.log(2)} train_acc={train_correct/train_steps} ', end='')
    print(f'test_length={args.test_length} test_ce={test_loss/test_steps/math.log(2)} test_acc={test_correct/test_steps}', flush=True)
