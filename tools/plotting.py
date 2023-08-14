from matplotlib import pyplot as plt
import collections
from IPython import display
from matplotlib_inline import backend_inline
import numpy as np
import inspect

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ProgressBoard(HyperParameters):
    """The board that plots data points in animation."""
    
    def __init__(self, epochs, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(5, 4), display=True):
        self.save_hyperparameters()
        self.fig = plt.figure(figsize=self.figsize)
        self.xlim = [0, epochs]
        self.ylim = [0, -100]
        self.xlabel = 'episode'

    def draw(self, x, y, label, every_n=1):
                        
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
        points = self.raw_points[label]
        line = self.data[label]
        
        #Populating points dictionary with latest point
        points.append(Point(x, y))
        
        #Drawing a point only every n steps
        if len(points) != every_n:
            return
        
        #Adding a point to the line 
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),mean([p.y for p in points])))
        points.clear()
        
        #Display the line
        useSvgDisplay()
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)
        
def useSvgDisplay():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def setFigsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    useSvgDisplay()
    plt.rcParams['figure.figsize'] = figsize

def setAxes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
