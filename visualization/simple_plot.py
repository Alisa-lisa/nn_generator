import matplotlib.pyplot as plt

def plot_predictions(timestamps, A, B):
    t = [i for i in range(len(A))]
    # predicted
    plt.plot(t, A, linewidth=3.0, color="blue")
    # actual
    plt.plot(t, B, linewidth=1.0, color="red")
    plt.show()