import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.models.attention import Attention


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)


class LocalFacialExtractor(nn.Module):
    def __init__(
            self,
            dim=1024,
            depth=10,
            dim_head=64,
            heads=16,
            num_id_token=5,
            num_queries=32,
            output_dim=2048,
            ff_mult=4,
    ):
        """
        Initializes the LocalFacialExtractor class.

        Parameters:
        - dim (int): The dimensionality of latent features.
        - depth (int): Total number of PerceiverAttention and FeedForward layers.
        - dim_head (int): Dimensionality of each attention head.
        - heads (int): Number of attention heads.
        - num_id_token (int): Number of tokens used for identity features.
        - num_queries (int): Number of query tokens for the latent representation.
        - output_dim (int): Output dimension after projection.
        - ff_mult (int): Multiplier for the feed-forward network hidden dimension.
        """
        super().__init__()

        # Storing identity token and query information
        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim ** -0.5

        # Learnable latent query embeddings
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        # Projection layer to map the latent output to the desired dimension
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        # Attention and FeedForward layer stack
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),  # Perceiver Attention layer
                        FeedForward(dim=dim, mult=ff_mult),  # FeedForward layer
                    ]
                )
            )

        # Mappings for each of the 5 different ViT features
        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ),
            )

        # Mapping for identity embedding vectors
        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y):
        """
        Forward pass for LocalFacialExtractor.

        Parameters:
        - x (Tensor): The input identity embedding tensor of shape (batch_size, 1280).
        - y (list of Tensor): A list of 5 visual feature tensors each of shape (batch_size, 1024).

        Returns:
        - Tensor: The extracted latent features of shape (batch_size, num_queries, output_dim).
        """

        # Repeat latent queries for the batch size
        latents = self.latents.repeat(x.size(0), 1, 1)

        # Map the identity embedding to tokens
        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token, self.dim)

        # Concatenate identity tokens with the latent queries
        latents = torch.cat((latents, x), dim=1)

        # Process each of the 5 visual feature inputs
        for i in range(5):
            vit_feature = getattr(self, f'mapping_{i}')(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)

            # Pass through the PerceiverAttention and FeedForward layers
            for attn, ff in self.layers[i * self.depth: (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        # Retain only the query latents
        latents = latents[:, :self.num_queries]
        # Project the latents to the output dimension
        latents = latents @ self.proj_out
        return latents
    

class PerceiverCrossAttention(nn.Module):
    """
    
    Args:
        dim (int): Dimension of the input latent and output. Default is 3072.
        dim_head (int): Dimension of each attention head. Default is 128.
        heads (int): Number of attention heads. Default is 16.
        kv_dim (int): Dimension of the key/value input, allowing flexible cross-attention. Default is 2048.
    
    Attributes:
        scale (float): Scaling factor used in dot-product attention for numerical stability.
        norm1 (nn.LayerNorm): Layer normalization applied to the input image features.
        norm2 (nn.LayerNorm): Layer normalization applied to the latent features.
        to_q (nn.Linear): Linear layer for projecting the latent features into queries.
        to_kv (nn.Linear): Linear layer for projecting the input features into keys and values.
        to_out (nn.Linear): Linear layer for outputting the final result after attention.

    """
    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        # Linear transformations to produce queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """

        Args:
            x (torch.Tensor): Input image features with shape (batch_size, n1, D), where:
                - batch_size (b): Number of samples in the batch.
                - n1: Sequence length (e.g., number of patches or tokens).
                - D: Feature dimension.
            
            latents (torch.Tensor): Latent feature representations with shape (batch_size, n2, D), where:
                - n2: Number of latent elements.
        
        Returns:
            torch.Tensor: Attention-modulated features with shape (batch_size, n2, D).
        
        """
        # Apply layer normalization to the input image and latent features
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        # Compute queries, keys, and values
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # Reshape tensors to split into attention heads
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        q_out = q.clone().detach()
        k_out = k.clone().detach()

        # Compute attention weights
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable scaling than post-division
        weight_out = weight.clone().detach()
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the output via weighted combination of values
        out = weight @ v

        # Reshape and permute to prepare for final linear transformation
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out) , weight_out,  q_out, k_out




class MultiIPRouter(nn.Module):
    def __init__(
        self,
        *,
        num_id_token=32,
        num_heads=16,
        inner_dim1=256,
        inner_dim2=128,
        inner_dim3=32,
        addtional_dim=3,
        num_layers=21,
        q_k_dim=2048,
    ):
        super().__init__()
       
        weight_dim = num_id_token * num_heads #512
        self.heads = num_heads

        self.norm = nn.LayerNorm(num_id_token * num_heads)
        self.norm_q = nn.LayerNorm(q_k_dim)
        self.norm_k = nn.LayerNorm(q_k_dim)
        self.to_q = nn.ModuleList([nn.Linear(q_k_dim, q_k_dim, bias=False) for _ in range(num_layers)])
        self.to_k = nn.ModuleList([nn.Linear(q_k_dim, q_k_dim, bias=False) for _ in range(num_layers)])
        # self.id_merge = nn.ModuleList([nn.Linear(num_id_token, 1, bias=False) for _ in range(num_layers)])
        self.layer_merge = nn.ModuleList([nn.Sequential(
            nn.Linear(weight_dim+addtional_dim, inner_dim1, bias=True),# 1026->256
            nn.ReLU(),
            nn.Linear(inner_dim1, inner_dim2, bias=False),# 256->128
            nn.ReLU(),
        ) for _ in range(num_layers)])
        
        # 3D rope
        self.frames = 13  
        self.height = 45  
        self.width = 30   
        self.feat_dim = weight_dim  
        self.register_buffer('pos_emb', self._create_positional_embedding())
        
        # spatial-temporal attention
        num_attention_layers = 4
        self.spatial_temporal_layers = nn.ModuleList([
            SpatialTemporalAttentionBlock(
                dim=self.feat_dim,
                num_heads=8,
                mlp_ratio=1
            ) for _ in range(num_attention_layers)
        ])
        
        # output projection
        self.final_proj = nn.Sequential(
            nn.Linear(self.feat_dim, 1),
            nn.Sigmoid()
        )

    def _create_positional_embedding(self):
        """create 3D positional embedding"""
        # create time dimension positional embedding
        t_pos = torch.arange(self.frames).float()
        t_emb = t_pos.unsqueeze(-1) / torch.pow(10000, torch.arange(0, self.feat_dim//3, 2).float() / (self.feat_dim//3))
        t_emb = torch.stack([torch.sin(t_emb), torch.cos(t_emb)], dim=-1).flatten(-2)
        t_emb = t_emb.unsqueeze(1).unsqueeze(1).expand(-1, self.height, self.width, -1)  # [13, 45, 30, feat_dim//3]

        # create height dimension positional embedding
        h_pos = torch.arange(self.height).float()
        h_emb = h_pos.unsqueeze(-1) / torch.pow(10000, torch.arange(0, self.feat_dim//3, 2).float() / (self.feat_dim//3))
        h_emb = torch.stack([torch.sin(h_emb), torch.cos(h_emb)], dim=-1).flatten(-2)
        h_emb = h_emb.unsqueeze(0).unsqueeze(2).expand(self.frames, -1, self.width, -1)  # [13, 45, 30, feat_dim//3]

        # create width dimension positional embedding
        w_pos = torch.arange(self.width).float()
        w_emb = w_pos.unsqueeze(-1) / torch.pow(10000, torch.arange(0, self.feat_dim//3, 2).float() / (self.feat_dim//3))
        w_emb = torch.stack([torch.sin(w_emb), torch.cos(w_emb)], dim=-1).flatten(-2)
        w_emb = w_emb.unsqueeze(0).unsqueeze(1).expand(self.frames, self.height, -1, -1)  # [13, 45, 30, feat_dim//3]

        # merge three dimensions positional embedding
        pos_emb = torch.cat([t_emb, h_emb, w_emb], dim=-1)  # [13, 45, 30, feat_dim]
        
        # expand positional embedding to full feature dimension
        if pos_emb.size(-1) < self.feat_dim:
            padding = torch.zeros(self.frames, self.height, self.width, self.feat_dim - pos_emb.size(-1))
            pos_emb = torch.cat([pos_emb, padding], dim=-1)
        
        return pos_emb

    def forward(self, weight, q_out, k_out, layer_idx, is_teacher_forcing=False):
        """
        Args:
            weight.shape: torch.Size([2(num_id), 16, 17550, 32])
            q_out.shape: torch.Size([2(num_id), 16,17550, 128])
            k_out.shape: torch.Size([2(num_id), 16,32, 128])
            layer_idx (int): 当前transformer层的索引
            output.shape: torch.Size([1, 17550, 2])
        """
        num_id = q_out.size(0)

        q = q_out.permute(0,2,3,1) # torch.Size([2, 17550, 128, 16])
        q = q.reshape(q.size(0),q.size(1),-1) # torch.Size([2, 17550, 2048])
        k = k_out.permute(0,2,3,1) # torch.Size([2, 32, 128, 16])
        k = k.reshape(k.size(0),k.size(1),-1) # torch.Size([2, 32, 2048])

        q = self.norm_q(q)
        q = self.to_q[layer_idx](q) # torch.Size([2, 17550, 2048])
        k = self.norm_k(k)
        k = self.to_k[layer_idx](k) # torch.Size([2, 32, 2048])

        q = reshape_tensor(q, self.heads) # torch.Size([2, 16, 17550, 128])
        k = reshape_tensor(k, self.heads) # torch.Size([2, 16, 32, 128])

        q_k_weight = q @ k.transpose(-2, -1) # torch.Size([2, 16, 17550, 32])
        q_k_weight = q_k_weight.permute(0,2,3,1) # torch.Size([2, 17550, 32, 16])
        q_k_weight = q_k_weight.reshape(q_k_weight.size(0),q_k_weight.size(1),-1) # torch.Size([2, 17550, 512])
        
        # q_k_weight = torch.softmax(q_k_weight.float(), dim=0).type(q_k_weight.dtype) # torch.Size([2, 17550, 1])
        q_k_weight = self.norm(q_k_weight) # torch.Size([2, 17550, 512])

        # reshape to video frame format [B, T, H, W]
        q_k_weight = q_k_weight.reshape(num_id, self.frames, self.height, self.width, -1)# torch.Size([2, 13, 45, 30, 512])
        
        # add positional embedding
        q_k_weight = q_k_weight + self.pos_emb
        
        # apply spatial-temporal attention layer
        for layer in self.spatial_temporal_layers:
            q_k_weight = layer(q_k_weight) # torch.Size([2, 13, 45, 30, 512])
            # q_k_weight = torch.softmax(q_k_weight.float(), dim=0).type(q_k_weight.dtype)

            
        # project to final output
        q_k_weight = q_k_weight.reshape(num_id, -1, self.feat_dim)  # [2, 17550, 512]
        output = self.final_proj(q_k_weight)  # [2, 17550, 1]
        
        return output.permute(2,1,0)  # [1,17550,2]

    def save(self, path: str):
        """save router state dictionary"""
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        """load router state dictionary"""
        state_dict = torch.load(path, map_location=next(self.parameters()).device)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        print(f"Router Missing keys: {missing_keys}")
        print(f"Router Unexpected keys: {unexpected_keys}")
        return missing_keys, unexpected_keys

class SpatialTemporalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        
        # three attention layers use diffusers' Attention implementation
        self.spatial_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            bias=True,
            cross_attention_dim=None
        )
        
        self.temporal_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            bias=True,
            cross_attention_dim=None
        )
        
        self.multi_id_attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            bias=True,
            cross_attention_dim=None
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(dim)  # spatial
        self.norm2 = nn.LayerNorm(dim)  # temporal
        self.norm3 = nn.LayerNorm(dim)  # multi-ID
        self.norm4 = nn.LayerNorm(dim)  # FFN
        
        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        # x shape: [num_id, T, H, W, C] = [2, 13, 45, 30, 512]
        num_id, T, H, W, C = x.shape
        
        # 1. spatial self-attention
        # treat each time step and ID as a batch: [num_id*T, H*W, C]
        x_space = x.reshape(num_id*T, H*W, C)
        x_space = self.norm1(x_space)
        x = x + self.spatial_attn(x_space).reshape(num_id, T, H, W, C)
        
        # 2. temporal self-attention
        # treat each spatial position and ID as a batch: [num_id*H*W, T, C]
        x_temp = x.permute(0, 2, 3, 1, 4).reshape(num_id*H*W, T, C)
        x_temp = self.norm2(x_temp)
        x = x + self.temporal_attn(x_temp).reshape(num_id, H, W, T, C).permute(0, 3, 1, 2, 4)
        
        # 3. multi-ID self-attention
        # treat time and spatial positions as a batch: [H*W*T, num_id, C]
        x_id = x.permute(2, 3, 1, 0, 4).reshape(H*W*T, num_id, C)
        x_id = self.norm3(x_id)
        x = x + self.multi_id_attn(x_id).reshape(H, W, T, num_id, C).permute(3, 2, 0, 1, 4)
        
        # 4. FFN
        x = x + self.mlp(self.norm4(x.reshape(-1, C))).reshape(num_id, T, H, W, C)
        
        return x

