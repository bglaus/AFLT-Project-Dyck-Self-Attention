import os
import matplotlib.pyplot as plt

results_dir = 'results/'
for filename in os.listdir(results_dir):
    if filename.lower().endswith('.txt'):
        filename = os.path.join(results_dir, filename)
        params = filename.split('.')[0].split('_')
        [ _, num_epochs, num_layers, num_heads, length, depth] = params
        file = open(filename, 'r')
        lines = file.read()
        out = lines.split('stdout=')[1].split('\\n')[0:-1]
        out = list(map(lambda x: x.replace('\'', ''), out))
        out = list(map(lambda x: x.replace('\"', ''), out))

        all_train_length = [float(x.split(' ')[0].split('=')[1]) for x in out]
        all_train_ce = [float(x.split(' ')[1].split('=')[1]) for x in out]
        all_train_acc = [float(x.split(' ')[2].split('=')[1]) for x in out]
        all_test_ce = [float(x.split(' ')[3].split('=')[1]) for x in out]
        all_test_acc = [float(x.split(' ')[4].split('=')[1]) for x in out]

        save_name = filename.split('.')[0] + '.png'
        # create 2 subplots 
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle(f"train vs. test metrics, word length = {length}", y= 0.95) 
        ax1, ax2 = fig.subplots(2,1)
        # plot cross entropy development on the first
        epochs = [i+1 for i in range(0,len(all_train_length))]
        ax1.plot(epochs,all_train_ce, label='train')
        ax1.plot(epochs,all_test_ce, label='test')
        #ax1.set_title('cross entropy',  fontsize='small')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('cross entropy')
        ax1.set_xlim([0, int(num_epochs)])
        ax1.set_ylim([0, 1.05])
        ax1.legend()
        # plot accuracy development on the second
        ax2.plot(epochs,all_train_acc, label='train')
        ax2.plot(epochs,all_test_acc, label='test')
        #ax2.set_title('accuracy',  fontsize='small')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.set_ylim([0, 1.05])
        ax2.set_xlim([0, int(num_epochs)])
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_name)
        #plt.show()
