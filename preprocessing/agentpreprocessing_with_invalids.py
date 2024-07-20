# trajectory dataset handling None data
class AgentPreProcessing_with_Invalids(Dataset):
    def __init__(self, root_path, settings, subset, history_frames, future_frames):
        
        self.root_path = root_path
        self.settings = settings
        self.num_agents = self.settings['total_num']
        self.subset = subset
        self.directory = os.path.join(self.root_path, self.settings["dataset"], self.subset)
        self.sequences = os.listdir(self.directory)
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.minimum_history = history_frames
        self.minimum_future = future_frames
        self.skip_frames = 10            # Skip frame is set to 10, since frames go 10 by 10 in the provided dataset 
        self.start_frame = 0
        self.total_samples = 0           # Total number of samples, obtained by summing each sequence total number of samples 
        self.samples_per_sequence = []   # List containing total number of samples per sequence 
        self.processed_sequences = []    # List containing processed_sequences 
        self.threshold = 5               # Threshold for distance calculations
        self.previous_data = []          # List to store past data    
        self.future_data = []            # List to store future data
        self.valid_counts = []           # List to store the count of valid agents 
        self.all_prior_data = []         # List to store all prior data 
        self.all_future_data = []        # List to store all future data 
        self.all_valid_counts = []       # List to store all valid counts 
        self.valid_data_count = 0        # Count of valid data samples 
        self.invalid_data_count = 0      # Count of invalid data samples 
        
        processor_class = DataPreprocessor    # Reference to the data preprocessor class 
        for sequence_name in self.sequences:
            sequence_processor = processor_class(root_path, settings, sequence_name, subset)     # Create a data processor for each sequence 
            # calculate the number of samples in the sequence 
            sequence_sample_count = sequence_processor.num_fr + 1 - (self.minimum_history - 1) * self.skip_frames - self.minimum_future * self.skip_frames + 1
            self.total_samples += sequence_sample_count  # Update total samples count
            self.samples_per_sequence.append(sequence_sample_count)   # Store the sample count for this sequence 
            self.processed_sequences.append(sequence_processor)   # Store the procced sequence 
        
        self.sample_indices = list(range(self.total_samples))   # List of sample indices -> [0,1, ..., self.total_samples] 
        self.current_index = 0   # Initializing current index 
        self.samples_per_sequence = [(x + 9) // 10 * 10 for x in self.samples_per_sequence]    # Adjusting samples per sequence for routing to the nearest 10 

    def __len__(self):
        """ the length of the dataset is given by the total number of samples divided by 10, since frame skip is 10 """
        return self.total_samples // 10

    def locate_sequence_and_frame(self, index):
        """ locate_sequence_and_frame determines the sequence and the specific frame position within that sequence corresponding to the given dataset index by 
            calculating the cumulative position and adjusting for skipped frames """
        current_position = copy.copy(index) * self.skip_frames
        for seq_id, count in enumerate(self.samples_per_sequence):
            if current_position < count:
                frame_position = current_position + (self.minimum_history - 1) * self.skip_frames + self.processed_sequences[seq_id].init_frame
                return seq_id, frame_position
            current_position -= count
        raise ValueError('Index {} is out of range'.format(index))

    def reformat_data(self, data):
        """ reformat data reformats the input data into structured arrays for past and future movements. It preprocesses the data, handling cases 
            either fewer or more agents valid agents than the expected total. If the data is invalid, it falls back on the last valid data, ensuring 
            consistent input for further processing. The reformatted data is returned """
        if data is not None:
            self.valid_data_count += 1
            prior_data, upcoming_data, valids = [], [], []
            prior_movement = np.array(torch.stack(data['pre_motion_3D'], dim=0))
            future_movement = np.array(torch.stack(data['fut_motion_3D'], dim=0))
            agent_count = prior_movement.shape[0]

            if agent_count < self.num_agents:
                self.process_data_for_few_agents(agent_count, prior_movement, future_movement, prior_data, upcoming_data, valids)
            else:
                self.process_data_for_many_agents(agent_count, prior_movement, future_movement, prior_data, upcoming_data, valids)
        else:
            # Fallback to previous valid data
            prior_data, upcoming_data, valids = self.previous_data, self.future_data, self.valid_counts
            self.invalid_data_count += 1

        return prior_data, upcoming_data, valids

    def process_data_for_few_agents(self, agent_count, prior_movement, future_movement, prior_data, upcoming_data, valids):
        """ process_data_for_few_agents handles cases where the current number of valid agents present in the data is less than the expected total """
        for i in range(agent_count):
            temp = np.zeros((self.num_agents, prior_movement.shape[1], 2))
            temp[:agent_count] = prior_movement
            prior_data.append(temp[None])

            temp = np.zeros((self.num_agents, future_movement.shape[1], 2))
            temp[:agent_count] = future_movement
            upcoming_data.append(temp[None])
            valids.append(agent_count)
        self.previous_data = prior_data
        self.future_data = upcoming_data
        self.valid_counts = valids
        

    def process_data_for_many_agents(self, agent_count, prior_movement, future_movement, prior_data, upcoming_data, valids):
        """ process_data_for_many_agents handles cases where the current number of valid agents present in the data is greater than or equal to the expected total """
        for i in range(agent_count):
            distances = np.linalg.norm(prior_movement[:, -1] - prior_movement[i:i+1, -1], axis=-1)
            close_indices = np.sum((distances < self.threshold).astype(int))

            if close_indices < self.num_agents:
                temp = np.zeros((self.num_agents, prior_movement.shape[1], 2))
                neighbors_idx = np.argsort(distances)
                neighbors_idx = neighbors_idx[:close_indices]
                temp[:close_indices] = prior_movement[neighbors_idx]
                prior_data.append(temp[None])

                temp = np.zeros((self.num_agents, future_movement.shape[1], 2))
                neighbors_idx = neighbors_idx[:close_indices]
                temp[:close_indices] = future_movement[neighbors_idx]
                upcoming_data.append(temp[None])
                valids.append(close_indices)
            else:
                neighbors_idx = np.argsort(distances)
                assert neighbors_idx[0] == i
                neighbors_idx = neighbors_idx[:self.num_agents]
                temp = prior_movement[neighbors_idx]
                prior_data.append(temp[None])
                temp = future_movement[neighbors_idx]
                upcoming_data.append(temp[None])
                valids.append(self.num_agents)
        self.previous_data = prior_data
        self.future_data = upcoming_data
        self.valid_counts = valids

    
    def collect_all_data(self, pre_data, fut_data, num_valid):
        """ collec_att_data collects and stores all past and future movement data, along with the count of valid agents, by appending the provided data to the 
            corresponding class attributes """
        self.all_prior_data.append(pre_data)
        self.all_future_data.append(fut_data)
        self.all_valid_counts.extend(num_valid)
    
    def get_concatenated_data(self):
        """ get_concated_data concatenates all connected past and future movement dat, along with the count of valid agents, into single arrays. It returns these 
            concatenated arrays """
        if self.all_prior_data:
            all_past_data = np.concatenate(self.all_prior_data, axis=0)
        else:
            all_past_data = np.empty((0, self.num_agents, self.history_frames, 2))

        if self.all_future_data:
            all_future_data = np.concatenate(self.all_future_data, axis=0)
        else:
            all_future_data = np.empty((0, self.num_agents, self.future_frames, 2))

        all_valid_num = np.array(self.all_valid_counts)

        return all_past_data, all_future_data, all_valid_num


    def __getitem__(self, index):
        """ getitem retrieves and process data for a given index. It determines the sample index, using the current index. Then, it locates the corresponding sequence 
            and frame based on the sample index and retrives the data for the specified frame from the identified sequence. It reformates the retrieved data into prior and future 
            motion data and valid agent counts """
        
        sample_idx = self.sample_indices[self.current_index]
        sequence_id, frame = self.locate_sequence_and_frame(sample_idx)
        sequence = self.processed_sequences[sequence_id]
        self.current_index += 1

        data = sequence(frame)

        prepared_data = self.reformat_data(data)
        pre_data, fut_data, num_valid = prepared_data
        pre_data = np.array(pre_data, dtype=np.float32)
        fut_data = np.array(fut_data, dtype=np.float32)
        num_valid = np.array(num_valid)
        
        pre_data = pre_data.reshape(-1, self.num_agents, self.history_frames, 2)
        fut_data = fut_data.reshape(-1, self.num_agents, self.future_frames, 2)
        num_valid = num_valid.reshape(-1)

        self.collect_all_data(pre_data, fut_data, num_valid)

        return pre_data, fut_data, num_valid