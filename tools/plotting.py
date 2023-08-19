from matplotlib import pyplot as plt
import collections
from IPython import display
from matplotlib_inline import backend_inline
from tools.utils import *


class ProgressBoard(HyperParameters):
    """The board that plots data points in animation."""
    
    def __init__(self, epochs, n, jupyter = False, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(5, 4), display=True):
        self.save_hyperparameters()
        self.fig = plt.figure(figsize=self.figsize)
        self.xlim = [0, epochs]
        self.ylim = [-200, 0]
        self.xlabel = 'episode'
        self.ylabel = 'reward'
        self.axes = plt.gca()
        self.setAxes(self.xlabel, self.ylabel, self.xlim, self.ylim, self.xscale, self.yscale)
        self.axes.grid()
        

    def draw(self, x, y, label, n = None):
        
        if n is None: n = self.n
                        
        #Creation of data structure for the points
        Point = collections.namedtuple('Point', ['x', 'y'])
        
        #ProgressBoard constructed for the first time
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        
        #Label used for the first time
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
            
        #Points will be populated with n number of points
        points = self.raw_points[label]
        
        #Line referring to the current label, containing a new edge every n points
        line = self.data[label]
        
        #Populating points dictionary with latest point
        points.append(Point(x, y))
        
        #Drawing a point only every n steps
        if len(points) != n: return
        
        #Adding a point to the line 
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),mean([p.y for p in points])))
        points.clear()
    
        #Effective Plotting
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            labels.append(k)
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],linestyle=ls, color=color, label = k)[0])
        self.axes.legend(plt_lines, labels)
              
        #Displaying the figure  
        if self.jupyter:
            useSvgDisplay()
            display.display(self.fig)
            display.clear_output(wait=True)
        else:
            plt.show(block = False)
            plt.pause((2*n) / self.epochs)
                
    def setAxes(self, xlabel, ylabel, xlim, ylim, xscale, yscale):
        """Set the axes for matplotlib."""
        self.axes.set_xlabel(xlabel), self.axes.set_ylabel(ylabel)
        self.axes.set_xscale(xscale), self.axes.set_yscale(yscale)
        self.axes.set_xlim(xlim),     self.axes.set_ylim(ylim)
        
    def block(self):
        plt.show(block = True)
        
def useSvgDisplay():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def setFigsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    useSvgDisplay()
    plt.rcParams['figure.figsize'] = figsize


    
