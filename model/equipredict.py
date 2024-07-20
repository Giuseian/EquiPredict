
class EquiPredict(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, input_dim, hidden_channel_dim, output_dim, device='cuda', act_fn=nn.SiLU(), layers=4, coords_weight=1.0, use_recurrent=False, normalize_diff=False, use_tanh=False, gnn_variant = 'GCN'):
        super(EquiPredict, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.layers = layers
        self.device = device 
        self.use_recurrent = use_recurrent

        self.node_embedding = nn.Linear(node_features, hidden_dim // 2)
        self.angle_embedding = nn.Linear(node_features, hidden_dim // 2)

        self.coord_transform = nn.Linear(input_dim, hidden_channel_dim, bias=False)
        self.velocity_transform = nn.Linear(input_dim, hidden_channel_dim, bias=False)

        self.use_dct = True
        self.validate_reasoning = True
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_categories = 4
        self.tao = 1

        self.given_category = False
        if not self.given_category:
            self.edge_network, self.coord_network, self.node_network, self.category_network = self.init_mlps(hidden_dim, hidden_channel_dim, act_fn)
        
        
        # Choose GCN VARIANT 
        self.gnn_variant = gnn_variant 
        if self.gnn_variant == 'GCN':
            self.gcl = GCNConv(self.hidden_dim, self.hidden_dim)
        elif self.gnn_variant == 'GAT':
            self.gcl = GATConv(self.hidden_dim, self.hidden_dim)        
        
        # Feature Learning Layers
        self.gcls = nn.ModuleList([self.create_gcl_layer(edge_features, hidden_dim, input_dim, hidden_channel_dim, output_dim, act_fn = nn.SiLU(), coords_weight = 1.0, recurrent = False, norm_diff = False, tanh = False) for _ in range(layers - 1)])

        # Prediction Heads
        self.predict_heads = nn.ModuleList([self.create_gcl_layer(edge_features, hidden_dim, input_dim, hidden_channel_dim, output_dim, act_fn = nn.SiLU(), coords_weight = 1.0, recurrent = False , norm_diff = False, tanh = False) for _ in range(20)])
        self.predict_heads_linear = nn.ModuleList([nn.Linear(hidden_channel_dim, output_dim, bias=False) for _ in range(20)])

        self.to(self.device)
    
    def init_mlps(self, hidden_dim, hidden_channel_dim, act_fn):
        """ init_mlp defined mlps that will be used later """
        edge_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_channel_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )

        coord_network = nn.Sequential(
            nn.Linear(hidden_channel_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_channel_dim * 2),
            act_fn
        )

        node_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )

        category_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_channel_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, self.num_categories),
            act_fn
        )

        return edge_network, coord_network, node_network, category_network
    
    def create_gcl_layer(self, in_edge_nf, hidden_nf, in_channel, hid_channel, out_channel, act_fn, coords_weight, recurrent, norm_diff, tanh):
        return Feature_learning_layer(hidden_nf, hidden_nf, hidden_nf, in_channel, hid_channel, out_channel, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU, recurrent=True, coords_weight=1.0, attention=True, norm_diff=False, tanh=False, apply_reasoning=False, input_reasoning=True, category_num=self.num_categories)
    
    def compute_dct_matrix(self, N, x):
        """ compute_dct_matrix compute the Discrete Cosine Transform (DCT) matrix and its inverse (IDCT). The DCT matrix is used to transform data into the frequency domain, while the IDCT matrix is used
            to transform data back from the frequency domain to the spatial domain"""
        dct_matrix = np.eye(N)
        for k in range(N):
            for i in range(N):
                weight = np.sqrt(2 / N)
                if k == 0:
                    weight = np.sqrt(1 / N)
                dct_matrix[k, i] = weight * np.cos(np.pi * (i + 0.5) * k / N)
        idct_matrix = np.linalg.inv(dct_matrix)
        dct_matrix = torch.from_numpy(dct_matrix).type_as(x)
        idct_matrix = torch.from_numpy(idct_matrix).type_as(x)
        return dct_matrix, idct_matrix
    
    def apply_dct(self, coords, vel, valid_agent_mask, agent_num, num_valid, batch_size):
        """ apply_dct applies Discrete Cosine Transform (DCT) to coordinates and velocities"""
        coords_center = torch.mean(coords * valid_agent_mask, dim=(1, 2), keepdim=True) * (agent_num / num_valid[:, None, None, None])
        coords -= coords_center
        dct_m, idct_m = self.compute_dct_matrix(self.input_dim, coords), self.compute_dct_matrix(self.output_dim, coords)
        dct_m, idct_m = dct_m[0].repeat(batch_size, agent_num, 1, 1), idct_m[1].repeat(batch_size, agent_num, 1, 1)
        coords, vel = torch.matmul(dct_m, coords), torch.matmul(dct_m, vel)
        return coords, coords_center, vel, idct_m
    

    def compute_interaction_categories(self, node_features, coords, valid_mask):
        """ compute_interaction_categories computes interaction categories between nodes based on their features and coordinates """
        
        batch_size, num_agents, _, _ = coords.shape
        node_features_1 = node_features[:, :, None, :].repeat(1,1,num_agents,1)
        node_features_2 = node_features[:, None, :, :].repeat(1,num_agents,1,1)
        
        # Calculate coordinate differences and distances
        coord_diff = coords[:, :, None, :, :] - coords[:, None, :, :, :]
        distances = torch.norm(coord_diff, dim=-1)
        distances = self.coord_network(distances)
        
        # Initialize edge features
        edge_features = self.message_passing(node_features_1, node_features_2, distances)

        # Compute interaction categories through message passing
        interaction_categories = self.message_aggregation(node_features, edge_features, distances, valid_mask, num_agents, batch_size)

        return interaction_categories

    def message_passing(self, node_features_1, node_features_2, distances):
        """ message_passing performs message passing to compute edge features using multi-head attention """
    
        edge_input = torch.cat([node_features_1, node_features_2, distances], dim=-1)
        
        # Apply multi-head attention
        multihead_attention = MultiHeadAttention(input_dim=edge_input.size(-1), hidden_dim=self.hidden_dim, device = self.device)
        edge_features = multihead_attention(edge_input, edge_input, edge_input)  # Self-attention
        

        return edge_features

    def message_aggregation(self, node_features, computed_edge_features, distances, valid_mask, num_agents, batch_size):
        """ message_aggregation aggregates edge features to update node representations and compute interaction categories """
        # Prepare mask to ignore self-loops
        mask = (torch.ones((num_agents, num_agents)) - torch.eye(num_agents)).type_as(computed_edge_features)
        mask = mask[None, :, :, None].repeat(batch_size, 1, 1, 1)

        # Aggregate edge features and update node representations
        updated_node_features = self.node_network(torch.cat([node_features, torch.sum(valid_mask * mask * computed_edge_features, dim=2)], dim=-1))

        # Prepare updated node features for interaction computation
        updated_node_features_1 = updated_node_features[:, :, None, :].repeat(1,1, num_agents,1)
        updated_node_features_2 = updated_node_features[:, None, :, :].repeat(1, num_agents, 1,1)
        updated_edge_input = torch.cat([updated_node_features_1, updated_node_features_2, distances], dim=-1)

        # Compute interaction categories
        interaction_categories = F.softmax(self.category_network(updated_edge_input) / self.tao, dim=-1)

        return interaction_categories

    def create_valid_mask(self, num_valid, num_agents):
        """ create_valid_mask create a mask to indicate valid interactions between agents in a 2D grid """
        batch_size = num_valid.shape[0]
        valid_mask = torch.zeros((batch_size, num_agents, num_agents))
        for i in range(batch_size):
            valid_mask[i, :num_valid[i], :num_valid[i]] = 1
        return valid_mask.unsqueeze(-1)

    def create_valid_mask2(self, num_valid, num_agents):
        """ create_valid_mask2 creates a mask to indicate valid agents in a 1D vector."""
        batch_size = num_valid.shape[0]
        valid_mask = torch.zeros((batch_size, num_agents))
        for i in range(batch_size):
            valid_mask[i, :num_valid[i]] = 1
        return valid_mask.unsqueeze(-1).unsqueeze(-1)

    def forward(self, node_features, coords, velocities, num_valid, edge_attr=None):
        """ forward method is the core of the EqMotion model, explained in details above """
        
        # Defining previous velocities, used to compute the cosine of the angle between them and the current velocity vectors 
        velocities_pre = torch.zeros_like(velocities)
        velocities_pre[:, :, 1:] = velocities[:, :, :-1]
        velocities_pre[:, :, 0] = velocities[:, :, 0]
        EPS = 1e-6
        vel_cosangle = torch.sum(velocities_pre * velocities, dim=-1) / ((torch.norm(velocities_pre, dim=-1) + EPS) * (torch.norm(velocities, dim=-1) + EPS))
        vel_angle = torch.acos(torch.clamp(vel_cosangle, -1, 1))

        batch_size, num_agents, _, _ = coords.shape

        valid_agent_mask = self.create_valid_mask2(num_valid, num_agents).type_as(node_features)   # It indicates which agents are valid in the current batch, helping in filtering out invalid data 

        # Applying DCT transform to coordinates and velocities to transform them into the frequency domain 
        if self.use_dct:
            coords, coords_center, velocities, idct_matrix = self.apply_dct(coords, velocities, valid_agent_mask, num_agents, num_valid, batch_size)

        # Creating embedding of node features and velocity angles using learned linear transformations 
        node_features = self.node_embedding(node_features)
        vel_angle_embedding = self.angle_embedding(vel_angle)
        # node_features is the feature vector that will be passed to the feature learning layer. It is a combination of the node features and the velocity angle embeddings 
        node_features = torch.cat([node_features, vel_angle_embedding], dim=-1)

        # Normalizing and transforming the coordinates and velocities to account for batch-wise variantions and prepare them for further preprocessing 
        coords_mean = torch.mean(torch.mean(coords * valid_agent_mask, dim=-2, keepdim=True), dim=-3, keepdim=True) * (num_agents / num_valid[:, None, None, None])
        coords = self.coord_transform((coords - coords_mean).transpose(2, 3)).transpose(2, 3) + coords_mean
        velocities = self.velocity_transform(velocities.transpose(2, 3)).transpose(2, 3)
        coord_velocity_combined = torch.cat([coords, velocities], dim=-2)

        valid_mask = self.create_valid_mask(num_valid, num_agents).type_as(node_features)
        
        # Determines the interaction categories for each edge. If categories are predefined, they are processed accordingly; otherwise, they are computed based on node features and edge attributes.
        category = F.one_hot(((edge_attr / 2) + 1).long(), num_classes=self.num_categories) if self.given_category else self.compute_interaction_categories(node_features, coord_velocity_combined, valid_mask)
        
        # Iteratively applying Feature Learning Layers to update the node features and coordinates based on the interaction categories
        category_per_layer = []
        for gcl in self.gcls:
            node_features, coords, _ = gcl(node_features, coords, velocities, valid_mask, valid_agent_mask, num_valid, category=category)
            category_per_layer.append(category)
        
        # Creating an index for all pairs of nodes to define edges and applies the final graph convolution layer to the node features using these edges 
        edge_index = torch.combinations(torch.arange(num_agents), r=2).t().to(self.device)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        node_features = self.gcl(node_features, edge_index)
        
        # If recurrent processing is enabled, node features are processed trough an LSTM layer to capture temporal dependencies
        if self.use_recurrent:
            node_features = node_features.view(batch_size * num_agents, -1, self.hidden_dim)
            node_features, _ = self.lstm(node_features)
            node_features = node_features.view(batch_size, num_agents, -1, self.hidden_dim)
        
        # Useing multiple prediction heads to generate outputs. Each head processes the node features to predict the coordinates, and the results are adjusted for mean and combined 
        all_out = []
        for i, (head, head_linear) in enumerate(zip(self.predict_heads, self.predict_heads_linear)):
            _, out, _ = head(node_features, coords, velocities, valid_mask, valid_agent_mask, num_valid, category=None)
            out_mean = torch.mean(torch.mean(out * valid_agent_mask, dim=-2, keepdim=True), dim=-3, keepdim=True) * (num_agents / num_valid[:, None, None, None])
            out = head_linear((out - out_mean).transpose(2, 3)).transpose(2, 3) + out_mean
            all_out.append(out[:, :, None, :, :])
        
        # Concatenating the outputs from all prediction heads and reshapes them to the final output format 
        coords = torch.cat(all_out, dim=2).view(batch_size, num_agents, 20, self.output_dim, -1)

        # If DCT was applied initially, it performs the inverse DCT to transform the coordinates back to the original domain 
        if self.use_dct:
            idct_matrix = idct_matrix[:, :, None, :, :]
            coords = torch.matmul(idct_matrix, coords)
            coords = coords + coords_center.unsqueeze(2)
        
        # Returning final predicted coordinates. If validate_reasoning is enabled, it also returns the interaction categories computed during the forward pass 
        if self.validate_reasoning:
            return coords, category_per_layer
        else:
            return coords, None 