#seeting up all the necessary parameters to configure the data processing pipeline
config = {
    'dataset': 'hotel',          # Name of the dataset
    'past_frames': 8,          # Number of past frames to consider
    'future_frames': 12,       # Number of future frames to consider
    'frame_skip': 10,          # Number of frames to skip between each step
    'min_past_frames': 8,      # Minimum number of past frames required for a valid ID
    'min_future_frames': 12,   # Minimum number of future frames required for a valid ID
    'traj_scale': 1,           # Scaling factor for trajectory coordinates
    'total_num': 3,            # Total number of frames to process
}