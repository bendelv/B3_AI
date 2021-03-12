import matplotlib.pyplot as plt
import numpy as np

list1 = [1, 3, 5]
list2 = [0.0, 0.4, 0.8, 1.0]

for i in list1:
    for j in list2:
        arr = np.load('Entropy{}_{}.npy'.format(i, j))
        x = np.arange(len(arr))
        plt.plot(x, arr, linewidth='1')

    plt.title('Evolution of entropy for w = {}'.format(i), fontsize = '15')
    plt.legend(['p = 0.0','p = 0.4', 'p = 0.8', 'p = 1.0'], fontsize = '10')
    plt.xlabel("Number of steps", fontsize='15')
    plt.ylabel("Entropy", fontsize='15')
    plt.savefig("GraphEntropy{}.png".format(i))
    plt.close()
