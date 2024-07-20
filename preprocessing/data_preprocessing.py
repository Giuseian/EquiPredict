import os
import numpy as np
import torch

class DataPreprocessor(object):
    def __init__(self, data_root, config, seq_name, split, log=None):
        '''Initialization of the DataPreprocessor class with essential configurations'''
        self.config = config #configuration settings
        self.data_root = data_root #root directory for dataset files
        self.split = split #indicates the sub-folder to choose (test, train, val)
        self.seq_name = seq_name #name of the file to process
        self.log = log #logger to debug 
        self.label_path = os.path.join(data_root, config['dataset'], self.split, seq_name) #construct the full path to the txt file to consider
        self.gt = self.load_ground_truth() #load ground truth from the specified label path
        self.xind, self.zind = 2, 3  #Index Positions for x and z coordinates in the dataset

    def load_ground_truth(self):
        '''This function reads the ground truth data from the file located at "self.label_path".
           It expects data to be tab-limited and attempts to lead it into a numpy array.'''
        #load ground truth data from a text file with tab as delimiter
        self.gt = np.genfromtxt(self.label_path, delimiter='\t', dtype=str)
        
        if self.gt.ndim == 1:
            #if data is not loaded as a 2D array, log a warning
            print(f"Warning: The data in {self.label_path} is not loaded as a 2D array.")
            print(f"Data: {self.gt}")
            
        self.gt = self.gt.astype('float32') #convert data type from numerical operations
        frames = self.gt[:, 0].astype(np.float32).astype(np.int_)  #extract and convert frame numbers
        fr_start, fr_end = frames.min(), frames.max()  #minimum and maximum among all frames
        self.init_frame = fr_start  #set the initial frame 
        self.num_fr = (fr_end + 1 - fr_start)    #get total number of frames
        return self.gt
        

    def get_id(self, data):
        """This function extracts and returns a list of unique IDs from the given data.
           It takes a 2D numpy array `data` as input, extracts the second column 
           (assumed to be IDs), and returns a copy of this column as a list."""
        return data[:, 1].copy().tolist()

    def filter_data_by_frame(self, frame, frame_skip, num_frames, past=False):
        '''Filters data to get subsequent corresponding specific frames
           This function generates a list of data arrays for frames around a given frame. 
           Depending on whether the past flag is set, it will collect frames before or after 
           the given frame. The frame indices are calculated based on the frame_skip and 
           num_frames parameters.'''
        data_list = [] #initialize empty list to hold the filtered data for each frame
        for i in range(num_frames):
            #get frame index based on whether we are looking at past or future frames
            frame_idx = frame - i * frame_skip if past else frame + (i+1) * frame_skip
            
            # Check if the calculated frame index is before the initial frame
            if frame_idx < self.init_frame:
                data = [] # If so, initialize an empty list for data
            #filter the ground truth data for the current frame index
            data = self.gt[self.gt[:, 0] == frame_idx]
            
            data_list.append(data) #add filtered data for the current frame to the data list
        return data_list #return list of filtered data arrays

    def get_valid_id(self, pre_data, fut_data):
        '''This function checks the IDs present in the past data frames (`pre_data`) and the future 
           data frames (`fut_data`). It verifies that each ID appears in a minimum number of past 
           frames and future frames as specified in the configuration. Only IDs that meet both 
           conditions are considered valid.'''
        cur_id = self.get_id(pre_data[0]) #extract current IDs from the first past data frame
        valid_id = [] #initialize an empty list to hold valid IDs
        for idx in cur_id:
            #check if the ID exists in the required number of past frames
            exist_pre = all(idx in data[:, 1] for data in pre_data[:self.config['min_past_frames']] if len(data) > 0)
            #check if the ID exists in the required number of future frames
            exist_fut = all(idx in data[:, 1] for data in fut_data[:self.config['min_future_frames']] if len(data) > 0)
            
            #if the ID exists in both past and future frames, add it to the valid_id list
            if exist_pre and exist_fut:
                valid_id.append(idx)
        return valid_id

    def compute_motion(self, data_tuple, valid_id, past=True):
        '''This function generates motion vectors and corresponding masks for each valid ID 
        in the data. It processes a series of frames, either past or future, depending on 
        the 'past' flag, and computes the motion for each ID by scaling the trajectory 
        coordinates. It handles missing data by carrying forward the last known data.'''
        frames = self.config['past_frames'] if past else self.config['future_frames'] #determine the # of frames to process based on "past" flag
        traj_scale = self.config['traj_scale'] #Trajectory scale factor for normalization
        motion = [] #Initialize a list to hold the motion vectors
        mask = [] #Initialize a list to hold the masks
        
        for identity in valid_id:
            mask_i = torch.zeros(frames)  #Initialize a tensor for the mask of the current ID
            box_3d = torch.zeros([frames, 2])  #Initialize a tensor for the motion vectors of the current ID
            for j in range(frames):
                data = data_tuple[j]
                if len(data) > 0 and identity in data[:, 1]:
                    #extract and scale the coordinates for the current ID
                    found_data = data[data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / traj_scale
                    if past:
                        box_3d[frames - 1 - j, :] = torch.from_numpy(found_data).float()
                        mask_i[frames - 1 - j] = 1.0
                    else:
                        box_3d[j, :] = torch.from_numpy(found_data).float()
                        mask_i[j] = 1.0
                elif j > 0:
                    #handle missing data by carrying forward the last known data
                    if past:
                        box_3d[frames - 1 - j, :] = box_3d[frames - j, :]
                    else:
                        box_3d[j, :] = box_3d[j - 1, :]
                else:
                    # Skip the case where the current ID is missing in the first frame
                    if past:
                        mask_i[frames - 1 - j] = 0.0
                        box_3d[frames - 1 - j, :] = torch.zeros(2)
                    else:
                        mask_i[j] = 0.0
                        box_3d[j,:] = torch.zeros(2)
            motion.append(box_3d)
            mask.append(mask_i)
        return motion, mask

    def __call__(self, frame):
        '''This method processes a given frame number to compute the motion data for both 
           past and future frames. It filters the data to get relevant frames, identifies 
           valid IDs that appear in both past and future data, and calculates the motion 
           vectors and masks for these IDs.'''
        #check if the frame is in the valid range
        if not (0 <= frame - self.init_frame < self.num_fr):
            raise ValueError(f'frame is {frame}, out of range')

        pre_data = self.filter_data_by_frame(frame, self.config['frame_skip'], self.config['past_frames'], past=True) #filter data to get past frames
        fut_data = self.filter_data_by_frame(frame, self.config['frame_skip'], self.config['future_frames'], past=False) #filter data to get future frames

        #identify valid IDs that appear in both past and future frames
        valid_id = self.get_valid_id(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or not valid_id:
            #print('None')
            return None #if there's no valid ID then return None
       
        pre_motion_3D, pre_motion_mask = self.compute_motion(pre_data, valid_id, past=True) #compute motion vectors and masks for past frames
        fut_motion_3D, fut_motion_mask = self.compute_motion(fut_data, valid_id, past=False) #compute motion vectors and masks for future frames
        
        #prepare data dictionary with all relevant information
        data = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            'pre_data': pre_data,
            'fut_data': fut_data,
            'valid_id': valid_id,
            'traj_scale': self.config['traj_scale'],
            'seq': self.seq_name,
            'frame': frame
        }
        
        #return stuctured data
        return data