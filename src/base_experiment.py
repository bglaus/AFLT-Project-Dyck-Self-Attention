import subprocess
import sys
from pathlib import Path

num_epochs = 150
num_layers = 2
num_heads = 2
train_lengths  = [5,10,25,50,75,100,150,200]
test_lengths = train_lengths
depth = 2

for length in train_lengths:
    command = ['python', 'dyck.py', '--epochs', f'{num_epochs}', '--layers', f'{num_layers}', '--heads', f'{num_heads}', '--train_length', f'{length}', '--test_length', f'{length}', '--depth', f'{depth}']
    foldername = 'results'
    Path(foldername).mkdir(parents=True, exist_ok=True)
    filename = f'{foldername}/result_{num_epochs}_{num_layers}_{num_heads}_{length}_{depth}.txt'
    sys.stdout = open(filename, 'w')
    # run bash command
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed:
        print(f'completed experiment {completed}')
        sys.stdout.close()