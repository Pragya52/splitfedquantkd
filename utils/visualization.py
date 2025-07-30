"""
Visualization utilities for federated learning training
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_curves(train_losses, val_losses, client_losses_history, save_path=None):
    """
    Plot comprehensive training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        client_losses_history: List of lists containing client losses
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall training and validation loss
    axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
    if val_losses:
        # Create x-axis for validation losses (assuming they're evaluated less frequently)
        val_x = np.linspace(0, len(train_losses)-1, len(val_losses))
        axes[0, 0].plot(val_x, val_losses, label='Validation Loss', linewidth=2, marker='o', markersize=4)
    
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Individual client losses
    colors = plt.cm.Set1(np.linspace(0, 1, len(client_losses_history)))
    for i, client_losses in enumerate(client_losses_history):
        if client_losses:
            axes[0, 1].plot(client_losses, label=f'Client {i+1}', color=colors[i], linewidth=2)
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Individual Client Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss distribution across clients (box plot)
    if any(client_losses_history):
        # Get recent losses for box plot
        recent_losses = []
        labels = []
        for i, client_losses in enumerate(client_losses_history):
            if client_losses:
                # Take last 20 losses or all if less than 20
                recent = client_losses[-20:] if len(client_losses) >= 20 else client_losses
                recent_losses.extend(recent)
                labels.extend([f'Client {i+1}'] * len(recent))
        
        if recent_losses:
            df = pd.DataFrame({'Loss': recent_losses, 'Client': labels})
            sns.boxplot(data=df, x='Client', y='Loss', ax=axes[1, 0])
            axes[1, 0].set_title('Loss Distribution Across Clients (Recent)')
            axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Loss convergence (moving average)
    if train_losses:
        window_size = min(10, len(train_losses) // 4)
        if window_size > 1:
            moving_avg = pd.Series(train_losses).rolling(window=window_size).mean()
            axes[1, 1].plot(train_losses, alpha=0.3, label='Original', color='blue')
            axes[1, 1].plot(moving_avg, label=f'Moving Average (window={window_size})', linewidth=2, color='red')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss Convergence')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_client_metrics(client_losses_history, save_path=None):
    """
    Plot detailed client-specific metrics
    
    Args:
        client_losses_history: List of lists containing client losses
        save_path: Path to save the plot
    """
    num_clients = len(client_losses_history)
    
    if num_clients == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Client loss trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, num_clients))
    for i, client_losses in enumerate(client_losses_history):
        if client_losses:
            axes[0, 0].plot(client_losses, label=f'Client {i+1}', color=colors[i], linewidth=2)
    
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Client Loss Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Final loss comparison
    final_losses = []
    client_names = []
    for i, client_losses in enumerate(client_losses_history):
        if client_losses:
            final_losses.append(client_losses[-1])
            client_names.append(f'Client {i+1}')
    
    if final_losses:
        bars = axes[0, 1].bar(client_names, final_losses, color=colors[:len(final_losses)])
        axes[0, 1].set_ylabel('Final Loss')
        axes[0, 1].set_title('Final Loss by Client')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_losses):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Loss improvement over time
    improvement_rates = []
    client_labels = []
    
    for i, client_losses in enumerate(client_losses_history):
        if len(client_losses) >= 10:
            # Calculate improvement as percentage decrease from first 10% to last 10%
            early_avg = np.mean(client_losses[:len(client_losses)//10 + 1])
            late_avg = np.mean(client_losses[-len(client_losses)//10:])
            improvement = (early_avg - late_avg) / early_avg * 100
            improvement_rates.append(improvement)
            client_labels.append(f'Client {i+1}')
    
    if improvement_rates:
        bars = axes[1, 0].bar(client_labels, improvement_rates, color=colors[:len(improvement_rates)])
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].set_title('Loss Improvement Rate by Client')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, improvement_rates):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + (0.5 if value >= 0 else -2),
                           f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top')
    
    # Plot 4: Client loss variance over time
    if any(len(losses) > 1 for losses in client_losses_history):
        variances = []
        rounds = []
        
        max_rounds = max(len(losses) for losses in client_losses
