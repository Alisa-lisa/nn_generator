import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import matplotlib.dates as mdates


def simple_plot_predictions(timestamps, A, B, state, time, save=False,
                            name="noname"):
    """
    Plots simple predictions vs actual data
    :param timestamps: time stamps of predictions
    :param A: predicted values
    :param B: actual values
    :param state: name of the y-axis
    :param time: name of the x-axis
    :param save: bool option to save the plot or not
    :param name: path where to save the plot
    :return: None, show plot
    """
    t = [datetime.fromtimestamp(float(i)) for i in timestamps]
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.plot(t, A, linewidth=2.0, color="blue")
    plt.plot(t, B, linewidth=1.0, color="red")

    xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    plt.ylabel(state)
    plt.yticks([0, 1])
    plt.xlabel(time)
    plt.xticks(rotation=90)

    red = mpatches.Patch(color='red', label='actual value')
    blue = mpatches.Patch(color='blue', label='predicted value')
    plt.legend(handles=[red, blue], bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save:
        plt.savefig(name)


def simple_cost_plot(iterations, cost, y_name, x_name, save=False,
                     name="noname"):
    """
    Plots cost function behaviour during training process
    :param iterations: current iteration
    :param cost: computed cost function
    :param y_name: y-axis label
    :param x_name: x-axis label
    :param save: bool to save the plot or not
    :param name: path where to save the file
    :return: None, plots cost function
    """
    plt.plot(iterations, cost, linewidth=2.0, color="blue")

    plt.ylabel(y_name)
    plt.xlabel(x_name)
    blue = mpatches.Patch(color='blue', label='cost function')
    plt.legend(handles=[blue], bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=2, mode="expand", borderaxespad=0.)
    if save:
        plt.savefig(name)

    plt.draw()
    plt.show(block=False)
