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
            LLaMADecoderLayer(config, layer_idx) 
            for layer_idx in range(1, 3)  # layers 1, 2
        ])
        
        # Trainable layers 31-32
        self.layers_31_32 = nn.ModuleList([
            LLaMADecoderLayer(config, layer_idx) 
            for layer_idx in range(31, 33)  # layers 31, 32
        ])
        
        # Trainable LLM head
        self.llm_head = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)
        
        # Layer normalization
        self.input_norm = RMSNorm(config.HIDDEN_DIM, eps=config.RMS_NORM_EPS)
        self.output_norm = RMSNorm(config.HIDDEN_DIM, eps=config.RMS_NORM_EPS)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward_layers_1_2(self, input_ids, attention_mask=None):
        """Forward pass through client layers 1-2"""
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.input_norm(hidden_states)
        
        # Process through layers 1-2
        for layer in self.layers_1_2:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        
        return hidden_states
    
    def forward_layers_31_32(self, hidden_states, attention_mask=None):
        """Forward pass through client layers 31-32"""
        # Process through layers 31-32
        for layer in self.layers_31_32:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        
        hidden_states = self.output_norm(hidden_states)
        return hidden_states
    
    def forward_final(self, hidden_states):
        """Generate final logits"""
        return self.llm_head(hidden_states)
    
    def forward(self, input_ids, attention_mask=None, server_processed_states=None, labels=None):
        """Complete forward pass for training"""
        if server_processed_states is None:
            # Standard forward pass (for standalone evaluation)
            hidden_states = self.forward_layers_1_2(input_ids, attention_mask)
            # Skip server processing for standalone mode
            hidden_states = self.forward_layers_31_32(hidden_states, attention_mask)
        else:
            # Federated forward pass
            hidden_states = self.forward_layers_31_32(server_processed_states, attention_mask)
        
        logits = self.forward_final(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.VOCAB_SIZE)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        return {
            'loss': loss,
            'logits': logits,
        }
