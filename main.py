"""
Main training script for Federated Learning with MedQuad dataset using LLaMA 2B
"""
import os
import sys
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import project modules
from config import Config
from models.llama_model import LLaMAForCausalLM
from models.federated_models import ServerModel, ClientModel
from data.dataset import MedQuadDataset, MedQuadDatasetQA, collate_fn
from data.non_iid_split import create_non_iid_split, calculate_non_iid_metrics
from utils.federated_utils import *
from utils.metrics import calculate_all_metrics, medical_specific_metrics, MetricsTracker, print_metrics_summary
from utils.visualization import plot_training_curves, plot_client_metrics, plot_communication_costs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FederatedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize tokenizer (using a simple tokenizer for demonstration)
        self.setup_tokenizer()
        
        # Initialize models
        self.setup_models()
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker()
        self.communication_tracker = CommunicationTracker()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.client_losses_history = [[] for _ in range(config.NUM_CLIENTS)]
        
    def setup_tokenizer(self):
        """Setup tokenizer"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Loaded DialoGPT tokenizer")
        except:
            # Fallback to simple tokenizer
            logger.warning("Could not load transformers tokenizer, using simple tokenizer")
            self.tokenizer = SimpleTokenizer(vocab_size=self.config.VOCAB_SIZE)
    
    def setup_models(self):
        """Initialize server and client models"""
        logger.info("Initializing models...")
        
        # Initialize server model
        self.server_model = ServerModel(self.config).to(self.device)
        
        # Initialize client models
        self.client_models = [
            ClientModel(self.config).to(self.device) 
            for _ in range(self.config.NUM_CLIENTS)
        ]
        
        # Log model sizes
        server_size = get_model_size(self.server_model)
        client_size = get_model_size(self.client_models[0])
        logger.info(f"Server model size: {server_size:.2f} MB")
        logger.info(f"Client model size: {client_size:.2f} MB")
        
    def setup_optimizers(self):
        """Initialize optimizers"""
        # Server optimizer (only for layers 3-30)
        self.server_optimizer = torch.optim.AdamW(
            [p for p in self.server_model.parameters() if p.requires_grad],
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Client optimizers
        self.client_optimizers = [
            torch.optim.AdamW(
                client.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            ) for client in self.client_models
        ]
        
        logger.info("Optimizers initialized")
    
    def load_and_split_data(self):
        """Load MedQuad dataset and create non-IID splits"""
        logger.info("Loading MedQuad dataset...")
        
        # Check if dataset exists
        if not os.path.exists(self.config.DATA_PATH):
            logger.error(f"Dataset not found at {self.config.DATA_PATH}")
            logger.info("Please run scripts/download_data.py first")
            sys.exit(1)
        
        # Load dataset
        df = pd.read_csv(self.config.DATA_PATH)
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Create dataset objects
        full_dataset = MedQuadDataset(
            df, 
            self.tokenizer, 
            max_length=self.config.MAX_LENGTH
        )
        
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")
        
        # Create non-IID splits for clients
        self.client_datasets = create_non_iid_split(
            train_dataset, 
            self.config.NUM_CLIENTS,
            alpha=self.config.DIRICHLET_ALPHA,
            min_samples_per_client=self.config.MIN_SAMPLES_PER_CLIENT
        )
        
        # Calculate non-IID metrics
        if hasattr(train_dataset, 'data'):
            labels = create_pseudo_labels(train_dataset.data)
        else:
            labels = create_pseudo_labels(train_dataset.dataset.data.iloc[train_dataset.indices])
        
        non_iid_metrics = calculate_non_iid_metrics(self.client_datasets, labels)
        logger.info(f"Non-IID metrics: {non_iid_metrics}")
        
        # Create data loaders
        self.client_loaders = [
            DataLoader(
                dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=2
            ) for dataset in self.client_datasets
        ]
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        logger.info("Data loading and splitting completed")
        
    def federated_training_step(self, round_num):
        """Single federated training step following the specified architecture"""
        round_losses = []
        
        # Set models to training mode
        self.server_model.train()
        for client_model in self.client_models:
            client_model.train()
        
        # Get data batches for all clients
        client_batches = []
        for i, loader in enumerate(self.client_loaders):
            try:
                batch = next(iter(loader))
                client_batches.append(prepare_batch_for_federated_training(batch, self.device))
            except StopIteration:
                # If a client runs out of data, cycle back
                loader_iter = iter(loader)
                batch = next(loader_iter)
                client_batches.append(prepare_batch_for_federated_training(batch, self.device))
        
        # STEP 1: CLIENT TO SERVER
        logger.debug("Step 1: Client to Server")
        client_hidden_states = []
        quantized_hidden_states = []
        
        for i in range(self.config.NUM_CLIENTS):
            batch = client_batches[i]
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # Forward pass through client layers 1-2
            with torch.no_grad():
                h_i = self.client_models[i].forward_layers_1_2(input_ids, attention_mask)
            
            # Record communication cost
            self.communication_tracker.record_upload(compute_communication_cost([h_i]))
            
            # Add Gaussian noise for privacy
            h_i_noise = add_gaussian_noise(h_i, self.config.SIGMA)
            
            # Quantize
            h_i_quantized = quantize(h_i_noise, self.config.K, self.config.QUANTIZATION_LEVELS)
            
            client_hidden_states.append(h_i)
            quantized_hidden_states.append(h_i_quantized)
        
        # STEP 2: SERVER PROCESSING AND TEACHER MODEL
        logger.debug("Step 2: Server Processing and Teacher Model")
        dequantized_states = []
        processed_states = []
        
        # Dequantize received states
        for i in range(self.config.NUM_CLIENTS):
            h_i_dequant = dequantize(
                quantized_hidden_states[i], 
                self.config.X_MAX, 
                self.config.X_MIN,
                self.config.QUANTIZATION_LEVELS
            )
            dequantized_states.append(h_i_dequant)
        
        # Process each client's data through server layers 3-30
        for i in range(self.config.NUM_CLIENTS):
            p_i = self.server_model.forward_layers_3_30(dequantized_states[i])
            processed_states.append(p_i)
        
        # Create ensemble for teacher model
        combined = torch.cat(dequantized_states, dim=-1)  # Concatenate along feature dimension
        teacher_logits = self.server_model.forward_ensemble(combined)
        
        # Generate soft targets for each client
        soft_targets = F.softmax(teacher_logits / self.config.TEMPERATURE, dim=-1)
        
        # Split soft targets among clients
        batch_size, seq_len, vocab_size = soft_targets.shape
        tokens_per_client = vocab_size // self.config.NUM_CLIENTS
        
        soft_targets_per_client = []
        for i in range(self.config.NUM_CLIENTS):
            start_idx = i * tokens_per_client
            end_idx = (i + 1) * tokens_per_client if i < self.config.NUM_CLIENTS - 1 else vocab_size
            soft_targets_i = soft_targets[..., start_idx:end_idx]
            soft_targets_per_client.append(soft_targets_i)
        
        # STEP 3: SERVER TO CLIENT
        logger.debug("Step 3: Server to Client")
        quantized_processed_states = []
        
        # Quantize processed states before sending back to clients
        for i in range(self.config.NUM_CLIENTS):
            p_i_quantized = quantize(processed_states[i], self.config.K, self.config.QUANTIZATION_LEVELS)
            quantized_processed_states.append(p_i_quantized)
            
            # Record communication cost
            self.communication_tracker.record_download(
                compute_communication_cost([p_i_quantized, soft_targets_per_client[i]])
            )
        
        # STEP 4: CLIENT FINAL PROCESSING AND LOSS
        logger.debug("Step 4: Client Final Processing and Loss")
        client_losses = []
        
        for i in range(self.config.NUM_CLIENTS):
            batch = client_batches[i]
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            
            # Dequantize received processed states
            p_i_dequant = dequantize(
                quantized_processed_states[i],
                self.config.X_MAX,
                self.config.X_MIN,
                self.config.QUANTIZATION_LEVELS
            )
            
            # Forward pass through client layers 31-32
            final_hidden_i = self.client_models[i].forward_layers_31_32(p_i_dequant, attention_mask)
            
            # Generate student logits
            student_logits_i = self.client_models[i].forward_final(final_hidden_i)
            
            # Compute losses
            # Task loss (cross-entropy with true labels)
            shift_logits = student_logits_i[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            l_task_i = loss_fct(shift_logits, shift_labels)
            
            # Knowledge distillation loss
            student_log_probs = F.log_softmax(student_logits_i / self.config.TEMPERATURE, dim=-1)
            
            # Adjust soft targets to match student logits dimensions
            if soft_targets_per_client[i].size(-1) != student_logits_i.size(-1):
                # Pad or truncate soft targets to match student logits
                target_size = student_logits_i.size(-1)
                current_size = soft_targets_per_client[i].size(-1)
                
                if current_size < target_size:
                    padding = torch.zeros(
                        soft_targets_per_client[i].size(0),
                        soft_targets_per_client[i].size(1),
                        target_size - current_size,
                        device=soft_targets_per_client[i].device
                    )
                    soft_targets_per_client[i] = torch.cat([soft_targets_per_client[i], padding], dim=-1)
                else:
                    soft_targets_per_client[i] = soft_targets_per_client[i][..., :target_size]
            
            l_KD_i = F.kl_div(
                student_log_probs, 
                soft_targets_per_client[i], 
                reduction='batchmean'
            )
            
            # Total loss
            l_total_i = l_task_i + self.config.ALPHA * l_KD_i
            client_losses.append(l_total_i)
            
            # Store individual client losses for tracking
            self.client_losses_history[i].append(l_total_i.item())
        
        # BACKPROPAGATION AND WEIGHT UPDATES
        self.backpropagation_and_update(client_losses)
        
        # Calculate average loss for this round
        avg_loss = sum(loss.item() for loss in client_losses) / len(client_losses)
        round_losses.append(avg_loss)
        
        return avg_loss
    
    def backpropagation_and_update(self, client_losses):
        """Perform backpropagation and update weights"""
        # Zero gradients
        self.server_optimizer.zero_grad()
        for optimizer in self.client_optimizers:
            optimizer.zero_grad()
        
        # Backpropagate client losses
        for i, loss in enumerate(client_losses):
            loss.backward(retain_graph=True)
            
            # Clip gradients
            clip_gradients(self.client_models[i], self.config.MAX_GRAD_NORM)
        
        # Update client parameters
        for optimizer in self.client_optimizers:
            optimizer.step()
        
        # Aggregate server gradients and update
        total_loss = sum(client_losses) / len(client_losses)
        total_loss.backward()
        
        # Clip server gradients
        clip_gradients(self.server_model, self.config.MAX_GRAD_NORM)
        
        # Update server parameters
        self.server_optimizer.step()
    
    def evaluate(self, round_num=None):
        """Evaluate the federated model"""
        logger.info("Starting evaluation...")
        
        # Set models to evaluation mode
        self.server_model.eval()
        for client_model in self.client_models:
            client_model.eval()
        
        all_predictions = []
        all_references = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = prepare_batch_for_federated_training(batch, self.device)
                
                # For evaluation, we'll use the first client model as representative
                # In practice, you might want to ensemble all clients
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                # Forward pass through the complete pipeline
                # Client layers 1-2
                h = self.client_models[0].forward_layers_1_2(input_ids, attention_mask)
                
                # Server layers 3-30
                p = self.server_model.forward_layers_3_30(h)
                
                # Client layers 31-32 and final prediction
                final_hidden = self.client_models[0].forward_layers_31_32(p, attention_mask)
                logits = self.client_models[0].forward_final(final_hidden)
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                loss = loss_fct(shift_logits, shift_labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Generate predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Convert to text for evaluation
                if hasattr(self.tokenizer, 'decode'):
                    pred_texts = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
                    ref_texts = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
                else:
                    # Simple conversion for basic tokenizer
                    pred_texts = [' '.join(map(str, pred.cpu().tolist())) for pred in predictions]
                    ref_texts = [' '.join(map(str, label.cpu().tolist())) for label in labels]
                
                all_predictions.extend(pred_texts)
                all_references.extend(ref_texts)
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        metrics = calculate_all_metrics(all_predictions, all_references)
        medical_metrics = medical_specific_metrics(all_predictions, all_references)
        
        # Combine all metrics
        all_metrics = {**metrics, **medical_metrics, 'val_loss': avg_loss}
        
        # Update tracking
        if round_num is not None:
            self.metrics_tracker.update(all_metrics, round_num)
            self.val_losses.append(avg_loss)
        
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        print_metrics_summary(all_metrics, f"Evaluation Results - Round {round_num}")
        
        return all_metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting federated training...")
        
        # Load and split data
        self.load_and_split_data()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for round_num in range(self.config.ROUNDS):
            logger.info(f"\n=== Round {round_num + 1}/{self.config.ROUNDS} ===")
            
            # New communication round
            self.communication_tracker.new_round()
            
            # Training step
            avg_loss = self.federated_training_step(round_num)
            self.train_losses.append(avg_loss)
            
            logger.info(f"Training Loss: {avg_loss:.4f}")
            
            # Evaluation
            if (round_num + 1) % self.config.EVAL_STEPS == 0:
                val_metrics = self.evaluate(round_num)
                val_loss = val_metrics['val_loss']
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(round_num, is_best=True)
                else:
                    patience_counter += 1
                
                # Check for early stopping
                if patience_counter >= 10:  # Patience of 10 evaluation periods
                    logger.info("Early stopping triggered")
                    break
            
            # Save checkpoint
            if (round_num + 1) % self.config.SAVE_STEPS == 0:
                self.save_checkpoint(round_num)
            
            # Plot progress
            if (round_num + 1) % self.config.LOGGING_STEPS == 0:
                self.plot_training_progress()
        
        # Final evaluation
        logger.info("\n=== Final Evaluation ===")
        final_metrics = self.evaluate()
        
        # Save training history and final plots
        self.save_training_history()
        self.plot_final_results()
        
        logger.info("Training completed!")
        return final_metrics
    
    def save_checkpoint(self, round_num, is_best=False):
        """Save model checkpoint"""
        checkpoint_name = f"checkpoint_round_{round_num}.pt"
        if is_best:
            checkpoint_name = "best_model.pt"
        
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, checkpoint_name)
        
        save_federated_checkpoint(
            self.server_model,
            self.client_models,
            round_num,
            [opt.state_dict() for opt in [self.server_optimizer] + self.client_optimizers],
            checkpoint_path
        )
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(15, 10))
        
        # Training loss
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(range(0, len(self.train_losses), len(self.train_losses)//len(self.val_losses)), 
                    self.val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Client losses
        plt.subplot(2, 3, 2)
        for i in range(self.config.NUM_CLIENTS):
            if self.client_losses_history[i]:
                plt.plot(self.client_losses_history[i], label=f'Client {i+1}')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Client Losses')
        plt.legend()
        plt.grid(True)
        
        # Metrics over time
        if self.metrics_tracker.history:
            key_metrics = ['exact_match', 'f1_token', 'bleu_1', 'rouge1']
            for idx, metric in enumerate(key_metrics):
                if metric in self.metrics_tracker.history:
                    plt.subplot(2, 3, 3 + idx)
                    values = [v for v in self.metrics_tracker.history[metric] if v is not None]
                    if values:
                        plt.plot(values)
                        plt.xlabel('Evaluation Round')
                        plt.ylabel(metric)
                        plt.title(f'{metric.upper()} Over Time')
                        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/plots/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_final_results(self):
        """Plot final results and statistics"""
        # Create comprehensive plots
        plot_training_curves(
            self.train_losses, 
            self.val_losses, 
            self.client_losses_history,
            save_path='results/plots/final_training_curves.png'
        )
        
        plot_client_metrics(
            self.client_losses_history,
            save_path='results/plots/client_metrics.png'
        )
        
        # Communication costs
        comm_stats = self.communication_tracker.get_stats()
        plot_communication_costs(
            comm_stats,
            save_path='results/plots/communication_costs.png'
        )
        
        logger.info("Final plots saved to results/plots/")
    
    def save_training_history(self):
        """Save training history to files"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'client_losses': self.client_losses_history,
            'metrics_history': self.metrics_tracker.history,
            'communication_stats': self.communication_tracker.get_stats(),
            'config': vars(self.config)
        }
        
        with open('results/training_history.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            history_serializable = convert_numpy(history)
            json.dump(history_serializable, f, indent=2)
        
        logger.info("Training history saved to results/training_history.json")

class SimpleTokenizer:
    """Simple tokenizer fallback"""
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.eos_token = '</s>'
        self.pad_token = '<pad>'
        
    def __call__(self, text, **kwargs):
        # Simple word-level tokenization
        words = text.split()
        ids = [hash(word) % self.vocab_size for word in words]
        
        max_length = kwargs.get('max_length', 512)
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        if kwargs.get('padding') == 'max_length':
            while len(ids) < max_length:
                ids.append(0)  # pad token id
        
        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([[1] * len(ids) + [0] * (max_length - len(ids))])
        }
    
    def decode(self, ids, **kwargs):
        return ' '.join([str(id) for id in ids if id != 0])

def main():
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    # Initialize configuration
    config = Config()
    
    # Initialize trainer
    trainer = FederatedTrainer(config)
    
    # Start training
    final_metrics = trainer.train()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print_metrics_summary(final_metrics, "Final Results")
    
    # Print communication statistics
    comm_stats = trainer.communication_tracker.get_stats()
    print(f"\nCommunication Statistics:")
    print(f"Total Upload: {comm_stats['total_upload_mb']:.2f} MB")
    print(f"Total Download: {comm_stats['total_download_mb']:.2f} MB")
    print(f"Total Communication: {comm_stats['total_communication_mb']:.2f} MB")

if __name__ == "__main__":
    main()
