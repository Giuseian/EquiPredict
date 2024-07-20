import argparse
import torch
import os
from torch import nn, optim
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import random
import sys 
from torch import nn
import torch.nn.functional as F
import math
import cv2
import glob
import copy
import warnings
import torch.nn.init as init
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv, GATConv 
import matplotlib.animation as animation
from IPython.display import HTML
import torch.optim as optim
import pickle
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
sys.path.append("..")


# Utility Functions
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

# Save Model Checkpoint
def save_checkpoint(epoch, model, optimizer, model_save_dir, subset, best=False):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    file_name = f"{subset}_ckpt_best.pth.tar" if best else f"{subset}_ckpt_{epoch}.pth.tar"
    file_path = os.path.join(model_save_dir, file_name)
    torch.save(state, file_path)
    
# Mask Function for Training
def get_valid_mask2(num_valid, agent_num):
    batch_size = num_valid.shape[0]
    valid_mask = torch.zeros((batch_size, agent_num))
    for i in range(batch_size):
        valid_mask[i, :num_valid[i]] = 1
    return valid_mask.unsqueeze(-1).unsqueeze(-1)

# Function to clear cache
def clear_cache():
    torch.cuda.empty_cache()
    
def load_and_plot_results(res_path):
    # Load the results from file
    with open(res_path, 'rb') as f:
        data = pickle.load(f)
    
    train_preds = data['results']['train_preds']
    train_gt = data['results']['train_gt']
    train_losses = data['results']['train_losses']
    val_losses = data['results']['val_losses']

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()

    return train_preds, train_gt, train_losses, val_losses 

# Creating animation to show differences between predictions and ground truth labels 
def create_animation(train_preds, train_gt, train_losses):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    plt.axis('off')

    def animate(i):
        ax.clear()
        preds = train_preds[i]
        gt = train_gt[i]
        x_preds = preds[0, 0, :, 0]   # Adjust indices as per your data shape
        y_preds = preds[0, 0, :, 1]
        x_gt = gt[0, 0, :, 0]
        y_gt = gt[0, 0, :, 1]
        ax.scatter(x_preds, y_preds, c='blue', label='Predictions', s=50)
        ax.scatter(x_gt, y_gt, c='red', label='Ground Truth', s=50)
        plt.title(f'Epoch {i} | Train Loss: {train_losses[i]:.5f}', fontsize=18, pad=20)
        plt.legend()
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    anim = animation.FuncAnimation(fig, animate, frames=len(train_losses), interval=800, repeat=True)
    html = HTML(anim.to_html5_video())
    plt.close()
    return html