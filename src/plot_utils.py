import numpy as np
import matplotlib.pyplot as plt

def plot_quadratic_function(func, xs=None, title=None):
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)

    Z = np.array([func(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    plt.contourf(X, Y, Z, levels=30)
    plt.colorbar()
    plt.title("Quadratic Function")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    if xs is not None:
        plt.scatter(xs[:, 0], xs[:, 1], color='red', marker='o', label="Points", s=8)
        plt.plot(xs[:, 0], xs[:, 1], color='red', linestyle='--', label="Dashed lines")

    if title:
        plt.savefig(f'figures/{title}.png', dpi=300)

    plt.show()
