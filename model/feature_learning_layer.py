from utility_functions import * 

class Feature_learning_layer(nn.Module):
    """Class for learning and updating features in an agent-based model"""
    def __init__(self, input_nf, output_nf, hidden_nf, input_c, hidden_c, output_c, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU, recurrent=True, coords_weight=1.0, attention=True, norm_diff=False, tanh=False, apply_reasoning=True, input_reasoning=False, category_num=2):
        super().__init__()
        # Flags for model behavior and feature processing configurations.
        self.recurrent = recurrent  # Determines if layer updates should be recurrent.
        self.attention = attention  # Determines if attention mechanisms should be applied.
        self.apply_reasoning = apply_reasoning  # Enables reasoning within the layer.
        self.category_num = category_num  # Number of output categories for classification tasks.

        # Core layers for feature processing.
        self.coord_vel = nn.Linear(2, 2, bias=False)  # Layer to update velocities.
        self.edge_mlp = self._build_mlp(input_nf * 2 + edges_in_d + hidden_c, hidden_nf, act_fn, layers=3)  # MLP for processing edge features.
        self.node_mlp = self._build_mlp(input_nf + hidden_nf + nodes_att_dim, output_nf, act_fn)  # MLP for node feature aggregation.
        self.category_mlps = nn.ModuleList([self._build_mlp(input_nf * 2 + hidden_c, hidden_c, act_fn, layers=3) for _ in range(category_num)])  # List of MLPs for different categories.
        
        # Attention mechanisms, instantiated if enabled.
        if attention:
            self.query = nn.Linear(hidden_c, hidden_c, bias=False)  # Generates query vectors for the attention mechanism.
            self.key = nn.Linear(hidden_c, hidden_c, bias=False)  # Generates key vectors for the attention mechanism.

        self.inner_attention_mlp = nn.Sequential(nn.Linear(hidden_nf, hidden_c), act_fn())  # MLP to process features within the attention mechanism.

    def build_mlp(self, input_size, output_size, act_fn, layers=2, add_tanh=False):
        """Function to dynamically build MLP structures"""
        mlp_layers = [nn.Linear(input_size, output_size), act_fn()]
        for _ in range(1, layers):
            mlp_layers.append(nn.Linear(output_size, output_size))
            mlp_layers.append(act_fn())
        if add_tanh:
            mlp_layers.append(nn.Tanh())  # Adds an optional Tanh layer for additional non-linearity.
        return nn.Sequential(*mlp_layers)

    def compute_edge_features(self, h, coord):
        """Computes edge features by considering spatial relationships and feature differences"""
        batch_size, agent_num, coord_dim, _ = coord.shape
        h1 = h.unsqueeze(2).expand(-1, -1, agent_num, -1)  # Expand features for each agent pair.
        h2 = h.unsqueeze(1).expand(-1, agent_num, -1, -1)  # Second expansion for pairwise comparison.
        coord_diff = coord.unsqueeze(2) - coord.unsqueeze(1)  # Compute pairwise coordinate differences.
        coord_dist = coord_diff.norm(dim=-1, keepdim=False)  # Calculate Euclidean distance between coordinates.
        edge_features = torch.cat((h1, h2, coord_dist), dim=-1)  # Concatenate features and distances.
        return self.edge_mlp(edge_features)  # Process concatenated features through an MLP.

    def update_coordinates(self, coord, edge_features):
        """Update agent coordinates based on computed edge features"""
        coord_factors = edge_features.mean(dim=2)  # Average features across dimensions.
        coord_factors = coord_factors.unsqueeze(-1).expand_as(coord)  # Expand features to match coordinate dimensions.
        return coord + coord_factors  # Update coordinates by adding feature-driven adjustments.

    def compute_node_features(self, h, edge_features, valid_mask):
        """Aggregate and refine node features using the edge features"""
        aggregated_edges = edge_features.sum(dim=2)  # Sum edge features to aggregate information.
        return self.node_mlp(torch.cat((h, aggregated_edges), dim=-1))  # Combine and process node and edge features.

    def apply_attention(self, coord, h, valid_mask_agent, num_valid):
        """Apply attention to refine feature adjustments based on their relevance"""
        query = self.query(h)  # Generate query vectors.
        key = self.key(h).transpose(1, 2)  # Generate and transpose key vectors.
        att_weights = torch.bmm(query, key)  # Compute raw attention weights.
        seq_len = valid_mask_agent.size(1)
        valid_mask_agent = valid_mask_agent.squeeze(-1).transpose(1, 2).expand(-1, seq_len, -1)  # Adjust and expand valid masks.
        att_weights = F.softmax(att_weights, dim=2) * valid_mask_agent  # Apply softmax and mask to normalize attention weights.
        coord_flattened = coord.reshape(coord.size(0), coord.size(1), -1)  # Flatten coordinates for matrix multiplication.
        coord_adjusted = torch.bmm(att_weights, coord_flattened)  # Adjust coordinates based on attention weights.
        coord_final = coord_adjusted.reshape(coord.size(0), coord.size(1), coord.size(2), coord.size(3))  # Reshape back to original dimensions.
        return coord_final  # Return adjusted coordinates.

    def forward(self, h, coord, vel, valid_mask, valid_mask_agent, num_valid, category=None):
        """Forward pass integrates all computations to update agent features and coordinates based on interactions and attention"""
        edge_features = self.compute_edge_features(h, coord)  # Compute edge features.
        coord = self.update_coordinates(coord, edge_features)  # Update coordinates based on edge features.
        coord += self.coord_vel(vel)  # Apply velocity updates.
        h = self.compute_node_features(h, edge_features, valid_mask)  # Compute and update node features.
        coord = self.apply_attention(coord, h, valid_mask_agent, num_valid)  # Refine coordinates using attention.
        return h, coord, category  # Return updated features and coordinates.
