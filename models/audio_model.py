"""
This module provides the implementation of an Audio Projection Model, which is designed for
audio processing tasks. The model takes audio embeddings as input and outputs context tokens
that can be used for various downstream applications, such as audio analysis or synthesis.

The AudioProjModel class is based on the ModelMixin class from the diffusers library, which
provides a foundation for building custom models. This implementation includes multiple linear
layers with ReLU activation functions and a LayerNorm for normalization.

Key Features:
- Audio embedding input with flexible sequence length and block structure.
- Multiple linear layers for feature transformation.
- ReLU activation for non-linear transformation.
- LayerNorm for stabilizing and speeding up training.
- Rearrangement of input embeddings to match the model's expected input shape.
- Customizable number of blocks, channels, and context tokens for adaptability.

The module is structured to be easily integrated into larger systems or used as a standalone
component for audio feature extraction and processing.

Classes:
- AudioProjModel: A class representing the audio projection model with configurable parameters.

Functions:
- (none)

Dependencies:
- torch: For tensor operations and neural network components.
- diffusers: For the ModelMixin base class.
- einops: For tensor rearrangement operations.

"""
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttentionProcessor
import torch
from diffusers import ModelMixin
from einops import rearrange
from torch import nn
from safetensors.torch import load_file



class AudioProjModel(torch.nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = torch.nn.LayerNorm(output_dim)
        
        self.conv1 = torch.nn.Conv1d(in_channels=context_tokens * output_dim,
                                     out_channels=context_tokens * output_dim,
                                     kernel_size=2,
                                     stride=2,
                                     padding=0)

    def forward(self, audio_embeds):
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        # context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f (m c)", f=video_length
        ) #torch.Size([2, 49, 24576])
        
        b, f, c = context_tokens.shape
        for _ in range(2):#torch.Size([2, 49, 24576])->torch.Size([2, 1+12, 24576])
            context_tokens = context_tokens.permute(0, 2, 1)
            if context_tokens.shape[-1] % 2 == 1:
                x_first, x_rest = context_tokens[..., 0], context_tokens[..., 1:]# x_first.shape torch.Size([2, 24576])
                if x_rest.shape[-1] > 0:
                    x_rest = self.conv1(x_rest)# x_first is not processed

                context_tokens = torch.cat([x_first[..., None], x_rest], dim=-1)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
            else:
                context_tokens = self.conv1(context_tokens)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
        
        context_tokens = rearrange(context_tokens, "b f (m c) -> b f m c", m=self.context_tokens) 
        context_tokens = self.norm(context_tokens)       

        return context_tokens
 

# Inject relative position bias in cross-attention
class AudioRelativePositionBias(nn.Module):
    def __init__(self, max_distance=8):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(2 * max_distance + 1))
        
    def forward(self, seq_len):
        # Generate relative position index matrix [-7, -6, ..., 0, ..., 6, 7]
        indices = torch.arange(seq_len) - torch.arange(seq_len).unsqueeze(-1)
        # Clip to [-7,7] and map to parameter indices [0, 1, ..., 15]
        clipped_indices = torch.clamp(indices + 7, 0, 15)
        return self.bias[clipped_indices]
    
class AudioAwareModel(nn.Module):
    def __init__(
        self,
        dim=3072,
        audio_dim=768,
        num_attention_heads=48,
        attention_head_dim=64,
        window_size=5,
        window_stride=1,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        num_layers=42, 
        audio_cross_attn_scale=0.05,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.window_stride = window_stride
        self.num_layers = num_layers
        self.learnable_scale = nn.Parameter(torch.tensor([0.01])) 
        self.audio_cross_attn_scale = audio_cross_attn_scale 
        self.audio_proj_model = AudioProjModel()
        # self.audio_relative_position_bias = AudioRelativePositionBias()
        
        
        self.layers = nn.ModuleList([
            self._build_single_layer(
                dim, audio_dim, num_attention_heads, attention_head_dim,
                norm_elementwise_affine, norm_eps
            ) for _ in range(num_layers)
        ])

        # zero cross attn
        # for layer in self.layers:  
        #     zero_module(layer['attn'])  
        # 

        # mute audio
        # self.init_mute_audio_feat()  
        self.mute_context_tokens = None
        self.mute_learnable_tokens  = nn.Parameter(torch.zeros(1,32,768))
        self.mute_dropout = nn.Dropout(0.1)


    def _build_single_layer(self, dim, audio_dim, heads, head_dim, affine, eps):
        return nn.ModuleDict({
            'norm_q': nn.LayerNorm(dim, eps, affine),
            # 'norm_kv': nn.LayerNorm(audio_dim, eps, affine),
            
            'attn': Attention(
                query_dim=dim,
                cross_attention_dim=audio_dim,
                dim_head=head_dim,
                heads=heads,
                bias=True,
            )
        })

    def sliding_windows(self, audio_embeds, hidden_states_num_frames):
        # The original number of video frames is 1 + f * 4; audio has four extra frames.
        assert 1+(hidden_states_num_frames-1) * 4 + (self.window_size - self.window_stride) == audio_embeds.shape[1], f"hidden_states_num_frames: {hidden_states_num_frames}, window_size: {self.window_size}, window_stride: {self.window_stride}, audio_embeds.shape[1]: {audio_embeds.shape[1]}"
        
        audio_embeds = audio_embeds.unfold(1, self.window_size, self.window_stride).permute(0,1,4,2,3) #(b,49,5,12,768)
        return audio_embeds


    def proj_in(self, audio_embeds):
        # (bs,f,8,12,768)->(bs,f,32,768)
        return self.audio_proj_model(audio_embeds)
        

    def _init_mute_audio_feat(self, cur_audio_context_tokens,num_frames):
        # return (1,f,32,768)
        mute_ae=torch.load("tests/input/ae_mute.pt")
        mute_ae=mute_ae[:(num_frames*4+1),:,:] #torch.Size([53, 12, 768])
        mute_ae = mute_ae.to(cur_audio_context_tokens.device, dtype=cur_audio_context_tokens.dtype)
        mute_ae = mute_ae.unsqueeze(0) #torch.Size([b, 53, 12, 768])
        mute_ae = self.sliding_windows(mute_ae, num_frames).contiguous() #torch.Size([b, 49, 5, 12, 768])
        mute_context_tokens = self.proj_in(mute_ae) #torch.Size([b, 13, 32, 768])
        assert cur_audio_context_tokens.shape == mute_context_tokens.shape, f"cur_audio_context_tokens.shape: {cur_audio_context_tokens.shape}, mute_context_tokens.shape: {mute_context_tokens.shape}"
        self.mute_context_tokens = mute_context_tokens
        return
    
    def get_mute_audio_feat(self, cur_audio_context_tokens,num_frames):
        # cur_audio_context_tokens:(1,f,32,768)
        if self.mute_context_tokens is None:
            self._init_mute_audio_feat(cur_audio_context_tokens,num_frames)
        learnable_tokens = self.mute_learnable_tokens.repeat(num_frames,1,1) #(f,32,768)
        learnable_tokens = self.mute_dropout(learnable_tokens) #(f,32,768)
        learnable_tokens = learnable_tokens.unsqueeze(0) #(1,f,32,768)
        return self.mute_context_tokens+learnable_tokens #(1,f,32,768)
    
        

    def forward(self, audio_embeds, hidden_states,num_frames,layer_index,mask=None):
        audio_context_tokens = audio_embeds

        layer = self.layers[layer_index]

        # First, reshape hidden states to (bs*f, hw, dim)
        # Reshape audio_context_tokens to (bs*f, 32, audio_dim)
        # Reshape hidden states: (B, F*HW, D) -> (B*F, HW, D)
        batch_size, seq_len, dim = hidden_states.shape
        single_frame_seq_len = seq_len // num_frames
        reshaped_hidden = hidden_states.reshape(
            batch_size * num_frames, 
            single_frame_seq_len, 
            dim
        )
        
        # Reshape audio context: (B, F, CTX, AD) -> (B*F, CTX, AD)
        reshaped_audio = audio_context_tokens.reshape(
            batch_size * num_frames,
            -1,
            audio_context_tokens.shape[-1]
        )
        
        # norm
        # norm_hidden = reshaped_hidden
        norm_hidden = layer['norm_q'](reshaped_hidden)
        # norm_audio = layer['norm_kv'](reshaped_audio)
        norm_audio = reshaped_audio

        attn_output = layer['attn'](
                hidden_states=norm_hidden,
                encoder_hidden_states=norm_audio,
            )
        
        # Reshape back: (B*F, HW, D) -> (B, F*HW, D)
        attn_output = attn_output.reshape(batch_size, seq_len, dim)

        return attn_output
    
if __name__ == "__main__":
    audio_model = AudioAwareModel()

    f=3;hw=4
    ori_f = (f-1)*4+1
    hidden_states = torch.randn(2, 12, 3072) # f=3,h*w=4
    audio_embeds = torch.randn(2, ori_f+4, 12, 768)

    audio_embeds = audio_model.sliding_windows(audio_embeds, 3) #torch.Size([2, 3, 8, 12, 768])
    audio_context_tokens = audio_model.proj_in(audio_embeds) #torch.Size([2, 3, 32, 768])

    for i in range(42):
        hidden_states = audio_model(audio_context_tokens, hidden_states, f,i)
    
    print(hidden_states.shape) #torch.Size([2, 12, 3072])

def zero_module(module):
    """
    Zeroes out the parameters of a given module.

    Args:
        module (nn.Module): The module whose parameters need to be zeroed out.

    Returns:
        None.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

