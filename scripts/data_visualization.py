import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def data_visualization(tx, y):

    print("Plotting data...", end=" ")

    # Batch creation
    idx_batch = np.random.randint(tx.shape[0], size=int(0.02*tx.shape[0]))
    tx_batch = tx[idx_batch, :]
    y_batch = y[idx_batch]
    len_scatter = np.shape(tx_batch)[1]

    plt.figure(figsize=(60, 60), dpi=100, facecolor='w', edgecolor='k')

    index_scatter = 1
    colors = ["red", "blue"]

    for xfeature in range(len_scatter):
        for yfeature in range(len_scatter):
            plt.subplot(len_scatter, len_scatter, index_scatter)

            if xfeature != yfeature:
                plt.scatter(tx_batch[:, xfeature], tx_batch[:, yfeature], s=0.4,  c=y_batch, cmap=matplotlib.colors.ListedColormap(colors))
            else:
                feature_values = tx_batch[:, xfeature]
                plt.hist(feature_values[~np.isnan(feature_values)], color="red")

            plt.xticks(())
            plt.yticks(())
            index_scatter += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('scatterplot_COLOR.png', dpi=100)

    print("Saved.")
