import subprocess
import sys
from pathlib import Path

num_epochs = 150
num_layers = 2
num_heads = 2
train_lengths  = [5,10,25,50,75,100,150,200]
test_lengths = train_lengths
num_par = 1
depth = 3
shuffle = False

for length in train_lengths:
    command = ['python', 'dyck1d_exact.py', '--train', '--epochs', f'{num_epochs}', '--train_length', f'{length}', '--test_length', f'{length}', '--num_par', f'{num_par}',  '--depth', f'{depth}', '--shuffle', f'{shuffle}']
    foldername = 'resultsImpr13DyckTrain'
    Path(foldername).mkdir(parents=True, exist_ok=True)
    filename = f'{foldername}/result_{num_epochs}_{length}_{depth}_{num_par}_{shuffle}.txt'
    sys.stdout = open(filename, 'w')
    # run bash command
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed:
        print(f'completed experiment {completed}')
        sys.stdout.close()