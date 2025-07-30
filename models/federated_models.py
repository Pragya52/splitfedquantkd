"""
Federated Learning Models Implementation
Server and Client models for the federated architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .llama_model import LLaMADecoderLayer, RMSNorm

class ServerModel(nn.Module):
    """
    Server model containing:
    - Layers 3-30 (trainable)
    - Copy of layers 31-32 (frozen)
    - LLM head copy (frozen)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Trainable layers 3-30 (28 layers total)
        self.layers_3_30 = nn.ModuleList([
            LLaMADecoderLayer(config, layer_idx) 
            for layer_idx in range(3, 31)  # layers 3 to 30
        ])
        
        # Copy layers 31-32 (frozen)
        self.layers_31_32_copy = nn.ModuleList([
            LLaMADecoderLayer(config, layer_idx) 
            for layer_idx in range(31, 33)  # layers 31, 32
        ])
        
        # LLM head copy (frozen)
        self.llm_head_copy = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)
        
        # Freeze copy parameters
        for param in self.layers_31_32_copy.parameters():
            param.requires_grad = False
        for param in self.llm_head_copy.parameters():
            param.requires_grad = False
            
        # Layer normalization for ensemble processing
        self.ensemble_norm = RMSNorm(config.HIDDEN_DIM * config.NUM_CLIENTS, eps=config.RMS_NORM_EPS)
        self.ensemble_projection = nn.Linear(
            config.HIDDEN_DIM * config.NUM_CLIENTS, 
            config.HIDDEN_DIM, 
            bias=False
        )
        
    def forward_layers_3_30(self, hidden_states, attention_mask=None):
        """Forward pass through server layers 3-30"""
        for layer in self.layers_3_30:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        return hidden_states
    
    def forward_ensemble(self, combined_states, attention_mask=None):
        """Forward pass through ensemble teacher model"""
        # Normalize and project combined states
        normalized = self.ensemble_norm(combined_states)
        projected = self.ensemble_projection(normalized)
        
        # Process through server layers 3-30
        ensemble_middle = self.forward_layers_3_30(projected, attention_mask)
        
        # Process through frozen layers 31-32
        hidden_states = ensemble_middle
        for layer in self.layers_31_32_copy:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        
        # Generate teacher logits
        teacher_logits = self.llm_head_copy(hidden_states)
        return teacher_logits

class ClientModel(nn.Module):
    """
    Client model containing:
    - Layers 1-2 (trainable)
    - Layers 31-32 (trainable)
    - LLM head (trainable)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding layer
        self.embed_tokens = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_DIM, padding_idx=0)
        
        # Trainable layers 1-2
        self.layers_1_2 = nn.ModuleList([
            LLaMADecoder
