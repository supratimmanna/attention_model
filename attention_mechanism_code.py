## Attention Mechanism using dot product

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalDotAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: Tensor of shape (batch_size, seq_len, hidden_size)
            decoder_hidden: Tensor of shape (batch_size, hidden_size)
        Returns:
            context: Tensor of shape (batch_size, hidden_size)
            attention_weights: Tensor of shape (batch_size, seq_len)
        """
        # Calculate attention scores using dot product
        attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate the context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights

# Example usage:
batch_size = 3
seq_len = 5
hidden_size = 10

encoder_outputs = torch.rand((batch_size, seq_len, hidden_size))
decoder_hidden = torch.rand((batch_size, hidden_size))

attention_model = GlobalDotAttention(hidden_size)
context, attention_weights = attention_model(encoder_outputs, decoder_hidden)

print("Context shape:", context.shape)
print("Attention weights shape:", attention_weights.shape)


##########################################################################################################
##########################################################################################################


## Attention Mechanism using general approach
# (general attention mechanism uses a learned linear transformation to calculate the attention scores. )

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalGeneralAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalGeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: Tensor of shape (batch_size, seq_len, hidden_size)
            decoder_hidden: Tensor of shape (batch_size, hidden_size)
        Returns:
            context: Tensor of shape (batch_size, hidden_size)
            attention_weights: Tensor of shape (batch_size, seq_len)
        """
        # Apply linear transformation to decoder hidden state
        transformed_decoder_hidden = self.linear(decoder_hidden)

        # Calculate attention scores using dot product
        attention_scores = torch.bmm(encoder_outputs, transformed_decoder_hidden.unsqueeze(2)).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate the context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights

# Example usage:
batch_size = 3
seq_len = 5
hidden_size = 10

encoder_outputs = torch.rand((batch_size, seq_len, hidden_size))
decoder_hidden = torch.rand((batch_size, hidden_size))

attention_model = GlobalGeneralAttention(hidden_size)
context, attention_weights = attention_model(encoder_outputs, decoder_hidden)

print("Context shape:", context.shape)
print("Attention weights shape:", attention_weights.shape)




######################################################################################################
######################################################################################################


## Attention Mechanism using concatenate approach (concatenation involves concatenating the decoder
# hidden state with each encoder output, and then applying a linear layer followed 
#by a tanh activation to calculate attention scores.)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalConcatAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalConcatAttention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: Tensor of shape (batch_size, seq_len, hidden_size)
            decoder_hidden: Tensor of shape (batch_size, hidden_size)
        Returns:
            context: Tensor of shape (batch_size, hidden_size)
            attention_weights: Tensor of shape (batch_size, seq_len)
        """
        # Repeat the decoder hidden state for each time step
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        # Concatenate encoder outputs and repeated decoder hidden state
        concatenated_inputs = torch.cat([encoder_outputs, repeated_decoder_hidden], dim=2)

        # Apply linear layer and tanh activation to calculate attention scores
        attention_scores = self.linear(concatenated_inputs).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate the context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights

# Example usage:
batch_size = 3
seq_len = 5
hidden_size = 10

encoder_outputs = torch.rand((batch_size, seq_len, hidden_size))
decoder_hidden = torch.rand((batch_size, hidden_size))

attention_model = GlobalConcatAttention(hidden_size)
context, attention_weights = attention_model(encoder_outputs, decoder_hidden)

print("Context shape:", context.shape)
print("Attention weights shape:", attention_weights.shape)




##########################################################################################################
#######################################################################################################


## Location based global attention (location-based scoring typically involves using a linear layer
# to calculate attention scores based on the position of the input sequence elements.)


import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalLocationAttention(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(GlobalLocationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.linear = nn.Linear(hidden_size + max_length, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: Tensor of shape (batch_size, seq_len, hidden_size)
            decoder_hidden: Tensor of shape (batch_size, hidden_size)
        Returns:
            context: Tensor of shape (batch_size, hidden_size)
            attention_weights: Tensor of shape (batch_size, seq_len)
        """
        # Calculate position-based scores
        position_scores = torch.arange(0, self.max_length).unsqueeze(0).repeat(encoder_outputs.size(0), 1).float()
        position_scores = position_scores / self.max_length  # Normalize to [0, 1]

        # Repeat the decoder hidden state for each time step
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        # Concatenate position scores and encoder outputs
        concatenated_inputs = torch.cat([encoder_outputs, position_scores], dim=2)

        # Apply linear layer to calculate attention scores
        attention_scores = self.linear(concatenated_inputs).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate the context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights

# Example usage:
batch_size = 3
seq_len = 5
hidden_size = 10
max_length = 20

encoder_outputs = torch.rand((batch_size, seq_len, hidden_size))
decoder_hidden = torch.rand((batch_size, hidden_size))

attention_model = GlobalLocationAttention(hidden_size, max_length)
context, attention_weights = attention_model(encoder_outputs, decoder_hidden)

print("Context shape:", context.shape)
print("Attention weights shape:", attention_weights.shape)



##################################################################################################################################
##################################################################################################################################

## Example code of multi-head self attention ############## 


import torch
import torch.nn.functional as F

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.projection_q = torch.nn.Linear(embed_size, embed_size)
        self.projection_k = torch.nn.Linear(embed_size, embed_size)
        self.projection_v = torch.nn.Linear(embed_size, embed_size)
        self.fc_out = torch.nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, x, mask=None):
        N, seq_len, _ = x.shape

        # Linear transformations for queries, keys, and values for each head
        queries = self.projection_q(x).view(N, seq_len, self.num_heads, self.head_dim)
        keys = self.projection_k(x).view(N, seq_len, self.num_heads, self.head_dim)
        values = self.projection_v(x).view(N, seq_len, self.num_heads, self.head_dim)

        # Transpose to get dimensions (N, num_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate attention scores using dot product
        scores = torch.matmul(queries, keys.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores / torch.sqrt(self.head_dim), dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, values)

        # Transpose back and concatenate along the last dimension
        attended_values = attended_values.transpose(1, 2).contiguous().view(N, seq_len, self.num_heads * self.head_dim)

        # Linear transformation to produce the final output
        output = self.fc_out(attended_values)

        return output

# Example usage
if __name__ == "__main__":
    # Input sequence (batch_size, sequence_length, embed_size)
    input_sequence = torch.rand((2, 5, 16))

    # Create a multi-head self-attention module with an embedding size of 16 and 4 attention heads
    multihead_attention = MultiHeadSelfAttention(embed_size=16, num_heads=4)

    # Calculate multi-head self-attention
    output_sequence = multihead_attention(input_sequence)

    print("Input Sequence:")
    print(input_sequence)
    print("\nOutput Sequence:")
    print(output_sequence)
