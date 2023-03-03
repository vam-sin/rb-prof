import matplotlib.pyplot as plt 
import seaborn as sns

f = open('slurm-564.out', 'r')

loss_vals = []
loss_train = []
epochs = []

num = 1
for line in f:
    if 'valid loss' in line:
        loss_vals.append(float(line.split(' ')[len(line.split(' '))-1].replace('\n','')))
        epochs.append(num)
        num += 1
    if 'Epoch Train Loss' in line:
        loss_train.append(float(line.split(' ')[11])/1e+6)
        # epochs.append(num)
        # num += 1

print(loss_vals, loss_train)

sns.lineplot(x=epochs, y=loss_vals, label='Valid Loss', color='#f0932b')
sns.lineplot(x=epochs, y=loss_train, label='Train Loss', color='#6ab04c')
plt.legend()
plt.show()
plt.savefig("loss_curves/TF-Reg-Model-__2__0-losses.png", format="png")

'''
check the test loss as well
'''