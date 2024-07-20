""" MultiHead Attention processes input sequences through multiple parallel attention heads, each learning different  
aspects of the relationships between tokens. The results from each head are combined and projected back to the desired 
output dimension. This approach helps the model capture diverse features and dependencies in the input data """

from utility_functions import * 

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.device = device
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.linear_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        # Move all linear layers to device
        self.linear_layers.to(self.device)
        self.to(self.device)

    def forward(self, query, key, value, mask=None):
    
        batch_size, seq_length, _, _ = query.shape        
        
        # Move inputs to device
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        
        # Project inputs using ModuleList
        Q = self.linear_layers[0](query)
        K = self.linear_layers[1](key)
        V = self.linear_layers[2](value)
        
        # Split heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        attention_output = self.linear_layers[3](attention_output)
        attention_output = attention_output.view(batch_size, seq_length, seq_length, attention_output.size(2))
        
        return attention_output