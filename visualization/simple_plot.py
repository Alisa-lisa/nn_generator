import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def simple_plot_predictions(timestamps, A, B, state, time, save=False, name="noname"):


    t = [datetime.fromtimestamp(float(i)) for i in timestamps]
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.plot(t, A, linewidth=2.0, color="blue")
    plt.plot(t, B, linewidth=1.0, color="red")

    xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    plt.ylabel(state)
    plt.yticks([0,1])
    plt.xlabel(time)
    plt.xticks(rotation=90)

    red = mpatches.Patch(color='red', label='actual value')
    blue = mpatches.Patch(color='blue', label='predicted value')
    plt.legend(handles=[red, blue], bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=2, mode="expand", borderaxespad=0.)

    if save:
        plt.savefig(name)

    plt.show()
