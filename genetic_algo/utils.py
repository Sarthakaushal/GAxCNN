import numpy as np
import matplotlib.pyplot as plt

def plot_graph(df, feature):
    
    fig,ax = plt.subplots()
    ax.plot(df[feature].weight,label=feature)

    ax.set_xlabel("Generation Number")
    ax.set_ylabel(feature)
    ax.legend(loc='best')

def array_to_prob(a:np.array):
    return a/np.sum(a)