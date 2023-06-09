import random

def is_in_dyck1d(word, max_depth):
        valid_dyck1d = True
        count = 0
        for letter in word:
            if letter == 0:
                count += 1
            if letter == 1:
                count -= 1
            if count < 0 or (max_depth != -1 and count > max_depth):
                valid_dyck1d = False
        return valid_dyck1d and count == 0


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
    
    def generate_dyck (self, size, depth=-1, change_only_one_char=True):
        inp = []
        label = True
        stack = []
        current_size = 0

        # Grammar: S -> (_i S )_i | SS | empty
        # With probability p we open a new paranthesis
        while current_size < size:
            prob = random.random()
            if (prob < self.p or len(stack) == 0) and len(stack) != depth:
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

        # Generate the output and with a probability of 1 - q produce a word that is not in Dyck anymore 
        # (from the Dyck word already produced)
        prob = random.random()
        if prob > self.q:
            # Generate a completely random string, which is not in the language
            if not change_only_one_char and self.pair_num == 1:
                inp = [random.randrange(2) for i in range(size)]
                label = is_in_dyck1d(inp, depth)
                while(label):
                    inp = [random.randrange(2) for i in range(size)]
                    label = is_in_dyck1d(inp, depth)
             # Change one single char (as stated by Hahn) with probability 1 - q
            else:
                # Change the label to False
                label = False
                char_idx = random.randint(0, len(inp)-1)
                rest_vocab = self.vocabulary.copy()
                rest_vocab.remove(inp[char_idx])
                inp[char_idx] = random.choice(rest_vocab)

        return inp, label

    def generate_shuffle_dyck (self, size):
        inp = []
        label = True
        pool = []
        current_size = 0

        # Grammar: S -> (_i S )_i | SS | empty
        # With probability p we open a new paranthesis
        while current_size < size:
            prob = random.random()
            if prob < self.p or len(pool) == 0:
                chosen_pair = random.choice (self.pairs) # randomly pick one of the pairs.
                # Add the opening paranthesis to the inp sequence
                inp.append(chosen_pair[0])
                # Add the closing paranthesis to the pool so it gets closed later
                pool.append(chosen_pair[1])
                current_size += 2
            else:
                # Add a random closing bracket from the pool
                inp.append(pool.pop(random.randrange(len(pool))))
        
        # Add the remaining closing paranthesis to the inp sequence
        while len(pool) != 0:
            inp.append(pool.pop(random.randrange(len(pool))))

        # Generate the output and with a probability of 1 - q produce a word that is not in Dyck anymore 
        # (from the Dyck word already produced)
        prob = random.random()
        if prob > self.q:
            # Change the label to False
            label = False
            # Randomly choose one of the ways to be not in dyck anymore
            prob2 = random.random()
            # Change one single char with probability 1 - q
            if prob2 >= 0: # Set this to 0.5 if second code should also be reachable
                char_idx = random.randint(0, len(inp)-1)
                rest_vocab = self.vocabulary.copy()
                rest_vocab.remove(inp[char_idx])
                inp[char_idx] = random.choice(rest_vocab)
            # Change all opening brackets to cooresponding closing brackets and vice versa
            # This is unreachable, but still important, since it's an example, which cleary is not in Dyck
            # But a Transformer can learn that this is not Dyck and therefore this is not overcoming Hahn's stated problem
            else:
                for i in range(len(inp)):
                    if inp[i] % 2 == 0:
                        inp[i] += 1
                    else:
                        inp[i] -= 1

        return inp, label