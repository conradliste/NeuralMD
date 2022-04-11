import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.cm as cm
import os
from IPython.display import display, clear_output

def plot(ts, samples, xlabel, ylabel, title=''):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, marker='x', label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def to_np(x):
    return x.detach().cpu().numpy()

def get_batch(t, true_y, batch_time, data_size, batch_size, device):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def plot_trajectories(true_y, pred_y=None, save_path=None):
    '''
    Plots the trajectories of dynamical systems

    obs:
    times:
    trajs:
    save: path to save plot
    figsize: figure size
    '''
    fig = plt.figure(figsize=((10,8)))
    ax = fig.add_subplot(1, 1, 1) 
    ax.cla()
    ax.set_title('Phase Portrait')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.scatter(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], c='g', s=2.5, marker='D')
    if pred_y != None:
        ax.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
    if save_path != None:
        plt.savefig(save_path)
    plt.show()
    plt.pause(0.001)
    
    
    display(ax)
    clear_output(wait = True)