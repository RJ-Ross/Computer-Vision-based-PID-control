# https://pythonprogramming.net/live-graphs-matplotlib-tutorial/

# Reads CSV file every few milliseconds and plots the data in the animated plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

class Live_Plot_Displacement:
    def __init__(self, x_label, y_label, x_units, y_units):
        style.use('fivethirtyeight')
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.lines = []
        self.x_label = x_label
        self.y_label = y_label
        self.x_nits = x_units
        self.y_units = y_units

    def update_data(self, data):
        self.lines.append(data)

    def animate(self, i):
        graph_data = open('example_data.csv','r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        for line in lines:
            if len(line) > 1:
                x, y = line.split(',')
                xs.append(float(x))
                ys.append(float(y))
        self.ax1.clear()
        self.ax1.plot(xs, ys)

    def run(self):
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=1)
        plt.show()

plot =  Live_Plot_Displacement("a","a","b","b")
plot.run()

# for val in range(30):
#     plot.update_data(str(val)+","+str(30-val)+"\n")