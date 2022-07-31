import sys
import numpy as np
import torch
from collections import defaultdict
import random

sys.setrecursionlimit(5000)

all_pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]
all_letters = []
for elt in all_pairs:
	all_letters += elt

class DyckGenerator ():
    def __init__ (self, num_pairs, p, q):
        self.pair_num = num_pairs
        self.pairs = all_pairs [:num_pairs]
        self.vocabulary = all_letters [:2*num_pairs]
        self.n_letters = len (self.vocabulary)
        
        self.p = p
        self.q = q
    
    def generate (self, size):
        inp = []
        label = True
        stack = []
        current_size = 0

        # Grammar: S -> (_i S )_i | SS | empty
        # With probability p we open a new paranthesis
        while current_size <= size:
            prob = random.random()
            if prob < self.p or len(stack) == 0:
                chosen_pair = random.choice (self.pairs) # randomly pick one of the pairs.
                # Add the opening paranthesis to the inp sequence
                inp.append(chosen_pair[0])
                # Add the closing paranthesis to the stack so it gets closed later
                stack.append(chosen_pair[1])
                current_size += 2
            else:
                # Add the closing paranthesis to the inp sequence
                inp.append(stack.pop())
        
        # Add the remaining closing paranthesis to the inp sequence
        while len(stack) != 0:
            inp.append(stack.pop())

        # Generate the output and
        # with a probability of 1 - q change one char to something else (so it is not in Dyck anymore)
        prob = random.random()
        if prob > self.q:
            # Change the label to False
            label = False
            # Change one single char with probability 1 - q
            char_idx = random.randint(0, len(inp)-1)
            rest_vocab = self.vocabulary.copy()
            rest_vocab.remove(inp[char_idx])
            inp[char_idx] = random.choice(rest_vocab)

        return inp, label