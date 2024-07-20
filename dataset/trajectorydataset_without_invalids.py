import torch
import numpy as np
from torch.utils.data import Dataset

# Class for Trajectory Dataset without Invalids
class TrajectoryDataset_without_Invalids(Dataset):
    """ Defining TrajectoryDataset to use with AgentPreProcessing_without_Invalids """
    def __init__(self, dataset, valids, settings, history_frames, future_frames):
        self.dataset = dataset # Assign the dataset
        self.valids = valids # Number of valid data points
        
        # Process each valid data point in the dataset
        for i in range(self.valids):
            prior_data, upcoming_data, valid = dataset[i]
            
        # Concatenate past and future data from the dataset
        self.all_past_data, self.all_future_data, self.all_valid_num = dataset.get_concatenated_data()
        self.settings = settings # Configuration settings
        self.traj_scale = self.settings["traj_scale"] # Scaling factor for trajectory coordinates
        self.history_frames = history_frames # Number of past frames
        self.future_frames = future_frames # Number of future frames
        
        # Concatenate past and future data along the third axis and convert to tensor
        self.all_past_future = np.concatenate([self.all_past_data, self.all_future_data], axis=2)
        self.all_past_future = torch.Tensor(self.all_past_future)
        self.all_valid_num = torch.Tensor(self.all_valid_num)
        
    def __len__(self):
        return self.all_past_future.shape[0] # Return the total number of sequences

    def __getitem__(self, index):
        """ Returns past_seq, future_seq, number of agents """
        # Normalize sequence by trajectory scale
        seq = self.all_past_future[index] / self.traj_scale
        valid_num = self.all_valid_num[index] # Number of valid agents
        past_seq = seq[:, :self.history_frames] # Extract past sequence
        future_seq = seq[:, self.history_frames:] # Extract future sequence
        return past_seq, future_seq, valid_num # Return past sequence, future sequence, and valid number