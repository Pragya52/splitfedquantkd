"""
Utility functions for federated learning operations
"""
import torch
import torch.nn.functional as F
import numpy as np

def add_gaussian_noise(tensor, sigma):
    """
    Add Gaussian noise to tensor for privacy preservation
    
    Args:
        tensor: Input tensor
        sigma: Standard deviation of noise
    
    Returns:
        Noisy tensor
    """
    if sigma <= 0:
        return tensor
    
    noise = torch.normal(0, sigma, size=tensor.shape, device=tensor.device, dtype=tensor.dtype)
    return tensor + noise

def quantize(tensor, k=10, levels=16):
    """
    Quantize tensor to specified number of levels using sigmoid approximation
    
    Args:
        tensor: Input tensor to quantize
        k: Sharpness parameter for sigmoid
        levels: Number of quantization levels (default: 16)
    
    Returns:
        Quantized tensor
    """
    quantized = torch.zeros_like(tensor)
    
    for n in range(levels):
        lower_bound = n 
        upper_bound = n + 1
        
        # Use sigmoid approximation for smooth quantization
        lower_sig = torch.sigmoid(k * (tensor - lower_bound))
        upper_sig = torch.sigmoid(k * (tensor - upper_bound))
        
        quantized += (lower_sig - upper_sig)
    
    return quantized

def dequantize(quantized_tensor, x_max=1.0, x_min=-1.0, levels=16):
    """
    Dequantize tensor back to continuous values
    
    Args:
        quantized_tensor: Quantized tensor
        x_max: Maximum value for output range
        x_min: Minimum value for output range
        levels: Number of quantization levels
    
    Returns:
        Dequantized tensor
    """
    # Scale from [0, levels-1] to [x_min, x_max]
    scale = (x_max - x_min) / (levels - 1)
    return quantized_tensor * scale + x_min

def compute_quantization_gradient(tensor, k=10, levels=16):
    """
    Compute gradients for the quantization function
    
    Args:
        tensor: Input tensor
        k: Sharpness parameter
        levels: Number of quantization levels
    
    Returns:
        Gradient tensor
    """
    gradient = torch.zeros_like(tensor)
    
    for n in range(levels):
        lower_bound = n - 0.5
        upper_bound = n + 0.5
        
        # Derivative of sigmoid
        lower_sig = torch.sigmoid(k * (tensor - lower_bound))
        upper_sig = torch.sigmoid(k * (tensor - upper_bound))
        
        lower_grad = k * lower_sig * (1 - lower_sig)
        upper_grad = k * upper_sig * (1 - upper_sig)
        
        gradient += (lower_grad - upper_grad)
    
    return gradient

def federated_averaging(client_models, client_weights=None):
    """
    Perform federated averaging of client model parameters
    
    Args:
        client_models: List of client models
        client_weights: Weights for each client (default: equal weights)
    
    Returns:
        Averaged state dict
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    # Normalize weights
    client_weights = np.array(client_weights)
    client_weights = client_weights / client_weights.sum()
    
    # Get the first model's state dict as template
    avg_state_dict = {}
    first_state_dict = client_models[0].state_dict()
    
    for key in first_state_dict.keys():
        # Initialize with zeros
        avg_state_dict[key] = torch.zeros_like(first_state_dict[key])
        
        # Weighted average
        for i, model in enumerate(client_models):
            avg_state_dict[key] += client_weights[i] * model.state_dict()[key]
    
    return avg_state_dict

def aggregate_server_gradients(client_gradients, aggregation_method='mean'):
    """
    Aggregate gradients from multiple clients for server update
    
    Args:
        client_gradients: List of gradient tensors from clients
        aggregation_method: Method for aggregation ('mean', 'median', 'trimmed_mean')
    
    Returns:
        Aggregated gradients
    """
    if not client_gradients:
        return None
    
    stacked_gradients = torch.stack(client_gradients)
    
    if aggregation_method == 'mean':
        return torch.mean(stacked_gradients, dim=0)
    elif aggregation_method == 'median':
        return torch.median(stacked_gradients, dim=0)[0]
    elif aggregation_method == 'trimmed_mean':
        # Remove top and bottom 10% and take mean
        sorted_grads, _ = torch.sort(stacked_gradients, dim=0)
        n_clients = len(client_gradients)
        trim_size = max(1, n_clients // 10)
        trimmed_grads = sorted_grads[trim_size:-trim_size] if trim_size > 0 else sorted_grads
        return torch.mean(trimmed_grads, dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

def create_attention_mask(seq_length, device):
    """
    Create causal attention mask for transformer
    
    Args:
        seq_length: Length of sequence
        device: Device to create mask on
    
    Returns:
        Attention mask tensor
    """
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def compute_communication_cost(tensors):
    """
    Compute communication cost (number of parameters transmitted)
    
    Args:
        tensors: List of tensors to transmit
    
    Returns:
        Total number of parameters
    """
    total_params = 0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            total_params += tensor.numel()
        elif isinstance(tensor, (list, tuple)):
            total_params += sum(t.numel() for t in tensor if isinstance(t, torch.Tensor))
    return total_params

def apply_differential_privacy(tensor, epsilon, delta, sensitivity=1.0):
    """
    Apply differential privacy noise to tensor
    
    Args:
        tensor: Input tensor
        epsilon: Privacy parameter (smaller = more private)
        delta: Privacy parameter
        sensitivity: Sensitivity of the function
    
    Returns:
        Tensor with DP noise added
    """
    # Calculate noise scale for Gaussian mechanism
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    # Add Gaussian noise
    noise = torch.normal(0, noise_scale, size=tensor.shape, device=tensor.device, dtype=tensor.dtype)
    return tensor + noise

def compute_model_similarity(model1, model2):
    """
    Compute cosine similarity between two models
    
    Args:
        model1: First model
        model2: Second model
    
    Returns:
        Cosine similarity score
    """
    params1 = torch.cat([p.flatten() for p in model1.parameters()])
    params2 = torch.cat([p.flatten() for p in model2.parameters()])
    
    cosine_sim = F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))
    return cosine_sim.item()

def clip_gradients(model, max_norm):
    """
    Clip gradients to prevent exploding gradients
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm for clipping
    
    Returns:
        Total norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def get_model_size(model):
    """
    Get the size of a model in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

class CommunicationTracker:
    """
    Track communication costs during federated training
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_upload = 0
        self.total_download = 0
        self.round_uploads = []
        self.round_downloads = []
    
    def record_upload(self, size_bytes):
        self.total_upload += size_bytes
        if len(self.round_uploads) == 0 or len(self.round_uploads) < len(self.round_downloads):
            self.round_uploads.append(size_bytes)
        else:
            self.round_uploads[-1] += size_bytes
    
    def record_download(self, size_bytes):
        self.total_download += size_bytes
        if len(self.round_downloads) == 0:
            self.round_downloads.append(size_bytes)
        else:
            self.round_downloads[-1] += size_bytes
    
    def new_round(self):
        if len(self.round_uploads) == len(self.round_downloads):
            self.round_uploads.append(0)
        self.round_downloads.append(0)
    
    def get_stats(self):
        return {
            'total_upload_mb': self.total_upload / (1024**2),
            'total_download_mb': self.total_download / (1024**2),
            'total_communication_mb': (self.total_upload + self.total_download) / (1024**2),
            'avg_upload_per_round_mb': np.mean(self.round_uploads) / (1024**2) if self.round_uploads else 0,
            'avg_download_per_round_mb': np.mean(self.round_downloads) / (1024**2) if self.round_downloads else 0,
        }

def prepare_batch_for_federated_training(batch, device):
    """
    Prepare a batch for federated training
    
    Args:
        batch: Input batch from dataloader
        device: Target device
    
    Returns:
        Prepared batch
    """
    prepared_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared_batch[key] = value.to(device)
        else:
            prepared_batch[key] = value
    
    return prepared_batch

def save_federated_checkpoint(server_model, client_models, round_num, optimizer_states, save_path):
    """
    Save federated learning checkpoint
    
    Args:
        server_model: Server model
        client_models: List of client models
        round_num: Current round number
        optimizer_states: Optimizer states
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'round': round_num,
        'server_model_state_dict': server_model.state_dict(),
        'client_model_state_dicts': [client.state_dict() for client in client_models],
        'optimizer_states': optimizer_states,
    }
    
    torch.save(checkpoint, save_path)

def load_federated_checkpoint(checkpoint_path, server_model, client_models, optimizers):
    """
    Load federated learning checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        server_model: Server model
        client_models: List of client models
        optimizers: List of optimizers
    
    Returns:
        Round number to resume from
    """
    checkpoint = torch.load(checkpoint_path)
    
    server_model.load_state_dict(checkpoint['server_model_state_dict'])
    
    for i, client_model in enumerate(client_models):
        client_model.load_state_dict(checkpoint['client_model_state_dicts'][i])
    
    if 'optimizer_states' in checkpoint and optimizers:
        for i, optimizer in enumerate(optimizers):
            if i < len(checkpoint['optimizer_states']):
                optimizer.load_state_dict(checkpoint['optimizer_states'][i])
    
    return checkpoint['round']

class EarlyStopping:
    """
    Early stopping utility for federated learning
    """
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, score, model=None):
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if model and self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        
        return False

def compute_local_update_norm(old_params, new_params):
    """
    Compute the norm of local updates
    
    Args:
        old_params: Parameters before update
        new_params: Parameters after update
    
    Returns:
        L2 norm of the update
    """
    update_norm = 0.0
    for old_p, new_p in zip(old_params, new_params):
        update_norm += torch.norm(new_p - old_p).item() ** 2
    
    return np.sqrt(update_norm)
