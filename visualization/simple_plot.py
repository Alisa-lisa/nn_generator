import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from matplotlib.dates import DateFormatter

def plot_predictions(timestamps, A, B, state, time):
    red = mpatches.Patch(color='red', label='actual value')
    blue = mpatches.Patch(color='blue', label='predicted value')
    plt.legend(handles=[red, blue], bbox_to_anchor=(0., 1.02, 1., .102),
               loc=3, ncol=2, mode="expand", borderaxespad=0.)

    t = [datetime.fromtimestamp(float(i)) for i in timestamps]
    plt.plot(t, A, linewidth=2.0, color="blue")
    plt.plot(t, B, linewidth=1.0, color="red")

    plt.ylabel(state)
    plt.yticks([0,1])
    plt.xlabel(time)
    plt.xticks(rotation=90)


    plt.show()