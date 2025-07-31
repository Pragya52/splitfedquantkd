import os
import sys
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import warnings
import gc
import signal
import psutil
import subprocess

warnings.filterwarnings("ignore")

# Import project modules
from config import Config
from models.llama_model import LLaMAForCausalLM
from models.federated_models import ServerModel, ClientModel
from data.dataset import MedQuadDataset, MedQuadDatasetQA, collate_fn, prepare_batch_for_federated_training
from data.non_iid_split import (
    create_non_iid_split,
    calculate_non_iid_metrics,
    create_distributed_non_iid_split,
    calculate_distributed_non_iid_metrics
)
from utils.federated_utils import *
from utils.metrics import calculate_all_metrics, medical_specific_metrics, MetricsTracker, print_metrics_summary

# Safe imports with fallbacks
try:
    from utils.visualization import plot_training_curves, plot_client_metrics
    try:
        from utils.visualization import plot_communication_costs
    except ImportError:
        def plot_communication_costs(*args, **kwargs):
            print("Communication costs plot not available - skipping")
except ImportError:
    def plot_training_curves(*args, **kwargs):
        print("Training curves plot not available - skipping")
    def plot_client_metrics(*args, **kwargs):
        print("Client metrics plot not available - skipping")
    def plot_communication_costs(*args, **kwargs):
        print("Communication costs plot not available - skipping")

def setup_logging(rank):
    """Setup logging for each GPU process"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger(__name__)

def signal_handler(sig, frame, logger):
    """Handle termination signals with stack trace"""
    signal_name = signal.Signals(sig).name
    logger.error(f"Received signal {signal_name}. Cleaning up...")
    import traceback
    logger.error(f"Stack trace:\n{''.join(traceback.format_stack())}")
    cleanup_distributed()
    sys.exit(1)

def get_gpu_processes():
    """Get processes using GPUs via nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        processes = []
        for line in lines[1:]:  # Skip header
            pid, mem = line.split(',')
            processes.append((int(pid), mem.strip()))
        return processes
    except Exception as e:
        print(f"Failed to query nvidia-smi: {e}")
        return []

def setup_distributed(rank, world_size):
    """Setup distributed training with robust CUDA context management"""
    logger = setup_logging(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Check CUDA availability and driver version
    if not torch.cuda.is_available():
        logger.error(f"GPU {rank}: CUDA not available")
        raise RuntimeError(f"GPU {rank}: CUDA not available")
    
    try:
        driver_version = torch.cuda.get_driver_version()
        cuda_version = torch.version.cuda
        logger.info(f"GPU {rank}: CUDA {cuda_version}, Driver {driver_version}")
    except:
        logger.warning(f"GPU {rank}: Could not retrieve CUDA/driver version")
    
    # Verify GPU accessibility
    try:
        device_count = torch.cuda.device_count()
        if rank >= device_count:
            logger.error(f"GPU {rank}: Invalid device index (only {device_count} GPUs available)")
            raise RuntimeError(f"GPU {rank}: Invalid device index")
        props = torch.cuda.get_device_properties(rank)
        logger.info(f"GPU {rank}: {props.name}, Total Memory: {props.total_memory / (1024**3):.2f}GB")
    except Exception as e:
        logger.error(f"GPU {rank}: Failed to access GPU - {e}")
        raise RuntimeError(f"GPU {rank}: Failed to access GPU")
    
    # Test CUDA context with larger allocation
    try:
        torch.cuda.set_device(rank)
        test_tensor = torch.randn(1000000, device=f'cuda:{rank}')  # ~4MB
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(rank)
        torch.cuda.synchronize()
        logger.info(f"GPU {rank}: CUDA context initialized successfully")
    except RuntimeError as e:
        logger.error(f"GPU {rank}: Failed to initialize CUDA context: {e}")
        raise
    
    # Check for stale processes
    gpu_processes = get_gpu_processes()
    for pid, mem in gpu_processes:
        if pid != os.getpid():
            logger.warning(f"GPU {rank}: Process {pid} using {mem} on GPU {rank}")
    
    # Initialize process group with retries
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logger.info(f"GPU {rank}: Attempting to initialize process group (attempt {attempt+1}/{max_retries})")
            init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=7200)
            )
            torch.distributed.barrier(device_ids=[rank])
            logger.info(f"GPU {rank}: Process group initialized successfully")
            break
        except Exception as e:
            logger.warning(f"GPU {rank}: Process group init failed: {e}")
            if attempt < max_retries - 1:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            raise RuntimeError(f"GPU {rank}: Failed to initialize process group after {max_retries} attempts")

def cleanup_distributed():
    """Cleanup distributed training"""
    try:
        destroy_process_group()
    except:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except:
        pass
    gc.collect()

def create_pseudo_labels(data):
    """Generate pseudo-labels for the dataset"""
    if 'category' in data.columns:
        return data['category'].values
    elif 'label' in data.columns:
        return data['label'].values
    elif 'topic' in data.columns:
        return data['topic'].values
    else:
        return np.arange(len(data)) % 10

class MultiGPUFederatedTrainer:
    def __init__(self, config, rank, world_size, is_distributed=True):
        self.config = config
        self.rank = rank
        self.world_size = world_size if is_distributed else 1
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.is_distributed = is_distributed
        
        try:
            torch.cuda.set_per_process_memory_fraction(0.5, device=rank)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass
        
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        
        self.logger = setup_logging(rank)
        self.logger.info(f"GPU {rank}/{world_size}: Using device: {self.device}")
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, self.logger))
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, self.logger))
        
        self.check_gpu_memory()
        self.setup_tokenizer()
        
        self.server_model = None
        self.local_client_models = []
        self.server_optimizer = None
        self.local_client_optimizers = []
        self.scaler = GradScaler()
        
        if rank == 0:
            try:
                self.metrics_tracker = MetricsTracker()
                self.communication_tracker = CommunicationTracker()
            except:
                self.metrics_tracker = None
                self.communication_tracker = None
            self.train_losses = []
            self.val_losses = []
            self.client_losses_history = [[] for _ in range(config.NUM_CLIENTS)]

    def check_gpu_memory(self):
        """Check available GPU memory and warn if insufficient"""
        if torch.cuda.is_available():
            memory_total = torch.cuda.get_device_properties(self.rank).total_memory / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(self.rank) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(self.rank) / (1024**3)
            memory_free = memory_total - memory_allocated
            self.logger.info(f"GPU {self.rank}: {memory_free:.2f}GB free / {memory_total:.2f}GB total, {memory_reserved:.2f}GB reserved")
            self.logger.info(f"GPU {self.rank} memory summary: {torch.cuda.memory_summary(device=self.rank)}")
            if memory_free < 10.0:
                self.logger.warning(f"GPU {self.rank}: Low memory ({memory_free:.2f}GB free). May cause issues.")
        else:
            self.logger.error(f"GPU {self.rank}: CUDA not available")
            raise RuntimeError(f"GPU {self.rank}: CUDA not available")

    def setup_tokenizer(self):
        """Setup tokenizer"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                'microsoft/DialoGPT-medium',
                cache_dir=None
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.rank == 0:
                self.logger.info("Loaded DialoGPT tokenizer")
        except Exception as e:
            if self.rank == 0:
                self.logger.warning(f"Could not load DialoGPT tokenizer: {e}")
            self.tokenizer = SimpleTokenizer(vocab_size=self.config.VOCAB_SIZE)
            if self.rank == 0:
                self.logger.warning("Using simple tokenizer fallback")

    def setup_memory_efficient_models(self):
        """Initialize models with memory efficiency"""
        if self.rank == 0:
            self.logger.info("Initializing memory-efficient models...")
            self.logger.info(f"GPU {self.rank} memory before model init: {torch.cuda.memory_summary(device=self.rank)}")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Assign exactly one client per GPU
        total_clients = self.config.NUM_CLIENTS
        if total_clients != self.world_size and self.is_distributed:
            self.logger.warning(f"GPU {self.rank}: NUM_CLIENTS ({total_clients}) should equal world_size ({self.world_size}) for optimal GPU utilization")
        self.local_clients = 1
        self.client_start_idx = self.rank if self.is_distributed else 0
        
        if self.rank == 0:
            self.logger.info(f"GPU {self.rank}: handling 1 client (client {self.client_start_idx})")
        
        try:
            # Initialize server model with reduced memory footprint
            self.server_model = ServerModel(self.config).to(self.device)
            if hasattr(self.server_model, 'gradient_checkpointing_enable'):
                self.server_model.gradient_checkpointing_enable()
                self.logger.info(f"GPU {self.rank}: Enabled gradient checkpointing for server model")
            else:
                self.logger.warning(f"GPU {self.rank}: ServerModel does not support gradient checkpointing")
            
            if self.is_distributed:
                self.server_model = DDP(
                    self.server_model,
                    device_ids=[self.rank],
                    find_unused_parameters=False,
                    broadcast_buffers=False
                )
            torch.cuda.synchronize()
            
            # Initialize one client model per GPU
            self.local_client_models = []
            try:
                client_model = ClientModel(self.config).to(self.device)
                if hasattr(client_model, 'gradient_checkpointing_enable'):
                    client_model.gradient_checkpointing_enable()
                    self.logger.info(f"GPU {self.rank}: Enabled gradient checkpointing for client model")
                else:
                    self.logger.warning(f"GPU {self.rank}: ClientModel does not support gradient checkpointing")
                if self.is_distributed:
                    client_model = DDP(
                        client_model,
                        device_ids=[self.rank],
                        find_unused_parameters=False,
                        broadcast_buffers=False
                    )
                self.local_client_models.append(client_model)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as e:
                self.logger.error(f"GPU {self.rank}: Cannot fit client model due to OOM: {e}")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                raise
            
            if self.rank == 0:
                try:
                    server_size = get_model_size(self.server_model.module if self.is_distributed else self.server_model)
                    client_size = get_model_size(self.local_client_models[0].module if self.is_distributed else self.local_client_models[0]) if self.local_client_models else 0
                    self.logger.info(f"Server model size: {server_size:.2f} MB")
                    self.logger.info(f"Client model size: {client_size:.2f} MB")
                except:
                    pass
                self.logger.info(f"GPU {self.rank}: Successfully loaded 1 client model")
        
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"GPU {self.rank}: Out of memory during model initialization: {e}")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            raise

    def setup_optimizers(self):
        """Initialize optimizers"""
        self.server_optimizer = torch.optim.AdamW(
            [p for p in self.server_model.parameters() if p.requires_grad],
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            eps=1e-6
        )
        
        self.local_client_optimizers = [
            torch.optim.AdamW(
                client.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                eps=1e-6
            ) for client in self.local_client_models
        ]
        
        if self.rank == 0:
            self.logger.info("Optimizers initialized")

    def setup_models_and_optimizers(self):
        """Initialize models and optimizers"""
        self.setup_memory_efficient_models()
        self.setup_optimizers()

    def load_and_split_data(self):
        """Load MedQuad dataset and create distributed non-IID splits"""
        if self.rank == 0:
            self.logger.info("Loading MedQuad dataset...")
        
        if not os.path.exists(self.config.DATA_PATH):
            if self.rank == 0:
                self.logger.error(f"Dataset not found at {self.config.DATA_PATH}")
            sys.exit(1)
        
        df = pd.read_csv(self.config.DATA_PATH)
        if self.rank == 0:
            self.logger.info(f"Loaded dataset with {len(df)} samples")
        
        full_dataset = MedQuadDataset(
            df,
            self.tokenizer,
            max_length=self.config.MAX_LENGTH,
            rank=self.rank
        )
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        if self.rank == 0:
            self.logger.info(f"Train size: {train_size}, Validation size: {val_size}")
        
        if self.rank == 0:
            self.logger.info("Creating distributed non-IID splits...")
        
        self.local_client_datasets = create_distributed_non_iid_split(
            train_dataset,
            self.config.NUM_CLIENTS,
            self.rank,
            self.world_size,
            alpha=self.config.DIRICHLET_ALPHA,
            min_samples_per_client=self.config.MIN_SAMPLES_PER_CLIENT
        )
        
        actual_clients = min(len(self.local_client_datasets), len(self.local_client_models)) if self.is_distributed else self.config.NUM_CLIENTS
        self.local_client_datasets = self.local_client_datasets[:actual_clients] if self.is_distributed else self.local_client_datasets
        
        self.local_client_loaders = []
        for client_dataset in self.local_client_datasets:
            client_loader = DataLoader(
                client_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False
            )
            self.local_client_loaders.append(client_loader)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        if self.rank == 0:
            self.logger.info("Memory-efficient data loading completed")
            self.logger.info(f"GPU {self.rank}: Using {len(self.local_client_datasets)} client datasets")

    def federated_training_step(self, round_num):
        """Memory-efficient federated training step"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.server_model.train()
        for client_model in self.local_client_models:
            client_model.train()
        
        local_losses = []
        
        for local_idx, (client_model, client_loader) in enumerate(zip(self.local_client_models, self.local_client_loaders)):
            try:
                batch = next(iter(client_loader))
                batch = prepare_batch_for_federated_training(batch, self.device)
                global_client_idx = self.client_start_idx + local_idx if self.is_distributed else local_idx
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                self.logger.info(f"GPU {self.rank}: Client {global_client_idx} - Input shape: {input_ids.shape}, Attention mask: {attention_mask.shape}")
                
                with autocast():
                    h_i = client_model.module.forward_layers_1_2(input_ids, attention_mask) if self.is_distributed else client_model.forward_layers_1_2(input_ids, attention_mask)
                    self.logger.info(f"GPU {self.rank}: Client {global_client_idx} - h_i shape: {h_i.shape}")
                    h_i_noise = add_gaussian_noise(h_i, self.config.SIGMA)
                    h_i_quantized = quantize(h_i_noise, self.config.K, self.config.QUANTIZATION_LEVELS)
                    h_i_dequant = dequantize(h_i_quantized, self.config.X_MAX, self.config.X_MIN, self.config.QUANTIZATION_LEVELS)
                    p_i = self.server_model.module.forward_layers_3_30(h_i_dequant) if self.is_distributed else self.server_model.forward_layers_3_30(h_i_dequant)
                    self.logger.info(f"GPU {self.rank}: Client {global_client_idx} - p_i shape: {p_i.shape}")
                    combined = h_i_dequant
                    teacher_logits = self.server_model.module.forward_ensemble(combined) if self.is_distributed else self.server_model.forward_ensemble(combined)
                    self.logger.info(f"GPU {self.rank}: Client {global_client_idx} - teacher_logits shape: {teacher_logits.shape}")
                    soft_targets = F.softmax(teacher_logits / self.config.TEMPERATURE, dim=-1)
                    p_i_quantized = quantize(p_i, self.config.K, self.config.QUANTIZATION_LEVELS)
                    p_i_dequant = dequantize(p_i_quantized, self.config.X_MAX, self.config.X_MIN, self.config.QUANTIZATION_LEVELS)
                    final_hidden_i = client_model.module.forward_layers_31_32(p_i_dequant, attention_mask) if self.is_distributed else client_model.forward_layers_31_32(p_i_dequant, attention_mask)
                    self.logger.info(f"GPU {self.rank}: Client {global_client_idx} - final_hidden_i shape: {final_hidden_i.shape}")
                    student_logits_i = client_model.module.forward_final(final_hidden_i) if self.is_distributed else client_model.forward_final(final_hidden_i)
                    self.logger.info(f"GPU {self.rank}: Client {global_client_idx} - student_logits_i shape: {student_logits_i.shape}")
                    
                    shift_logits = student_logits_i[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    l_task_i = loss_fct(shift_logits, shift_labels)
                    student_log_probs = F.log_softmax(student_logits_i / self.config.TEMPERATURE, dim=-1)
                    if soft_targets.size(-1) != student_logits_i.size(-1):
                        if soft_targets.size(-1) > student_logits_i.size(-1):
                            soft_targets = soft_targets[..., :student_logits_i.size(-1)]
                        else:
                            padding_size = student_logits_i.size(-1) - soft_targets.size(-1)
                            padding = torch.zeros(*soft_targets.shape[:-1], padding_size, device=soft_targets.device)
                            soft_targets = torch.cat([soft_targets, padding], dim=-1)
                    l_KD_i = F.kl_div(student_log_probs, soft_targets, reduction='batchmean')
                    l_total_i = l_task_i + self.config.ALPHA * l_KD_i
                
                self.scaler.scale(l_total_i).backward()
                local_losses.append(l_total_i)
                if self.rank == 0:
                    self.client_losses_history[global_client_idx].append(l_total_i.item())
            except (StopIteration, torch.cuda.OutOfMemoryError) as e:
                if self.rank == 0:
                    self.logger.warning(f"GPU {self.rank}: Skipping client {local_idx} due to memory/data issue: {e}")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
        
        if local_losses:
            self.scaler.step(self.server_optimizer)
            for optimizer in self.local_client_optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()
            avg_local_loss = sum(loss.item() for loss in local_losses) / len(local_losses)
        else:
            avg_local_loss = 0.0
        
        if self.is_distributed:
            try:
                all_losses = [torch.tensor(0.0, device=self.device) for _ in range(self.world_size)]
                torch.distributed.all_gather(all_losses, torch.tensor(avg_local_loss, device=self.device))
                torch.distributed.barrier(device_ids=[self.rank])
                global_avg_loss = sum(loss.item() for loss in all_losses) / len(all_losses)
            except:
                global_avg_loss = avg_local_loss
        else:
            global_avg_loss = avg_local_loss
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return global_avg_loss

    def memory_efficient_backprop(self, local_losses):
        """Memory-efficient backpropagation"""
        self.server_optimizer.zero_grad()
        for optimizer in self.local_client_optimizers:
            optimizer.zero_grad()
        
        for i, loss in enumerate(local_losses):
            self.scaler.scale(loss).backward(retain_graph=(i < len(local_losses) - 1))
            if i < len(self.local_client_models):
                try:
                    clip_gradients(self.local_client_models[i], self.config.MAX_GRAD_NORM)
                except:
                    pass
        
        self.scaler.step(self.server_optimizer)
        for optimizer in self.local_client_optimizers:
            self.scaler.step(optimizer)
        self.scaler.update()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def evaluate(self, round_num=None):
        """Memory-efficient evaluation"""
        if self.rank != 0:
            return {}
        
        self.logger.info("Starting memory-efficient evaluation...")
        
        self.server_model.eval()
        for client_model in self.local_client_models:
            client_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                if batch_idx > 20:
                    break
                try:
                    batch = prepare_batch_for_federated_training(batch, self.device)
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    
                    if self.local_client_models:
                        client_model = self.local_client_models[0]
                        with autocast():
                            h = client_model.module.forward_layers_1_2(input_ids, attention_mask) if self.is_distributed else client_model.forward_layers_1_2(input_ids, attention_mask)
                            p = self.server_model.module.forward_layers_3_30(h) if self.is_distributed else self.server_model.forward_layers_3_30(h)
                            final_hidden = client_model.module.forward_layers_31_32(p, attention_mask) if self.is_distributed else client_model.forward_layers_31_32(p, attention_mask)
                            logits = client_model.module.forward_final(final_hidden) if self.is_distributed else client_model.forward_final(final_hidden)
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        loss_fct = nn.CrossEntropyLoss()
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.view(-1)
                        
                        loss = loss_fct(shift_logits, shift_labels)
                        total_loss += loss.item()
                        num_batches += 1
                except torch.cuda.OutOfMemoryError:
                    self.logger.warning("OOM during evaluation, continuing...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        metrics = {'val_loss': avg_loss}
        
        if round_num is not None:
            if self.metrics_tracker:
                self.metrics_tracker.update(metrics, round_num)
            self.val_losses.append(avg_loss)
        
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        return metrics

    def train(self):
        """Main memory-efficient training loop"""
        if self.rank == 0:
            self.logger.info("Starting memory-efficient distributed federated training...")
        
        self.load_and_split_data()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for round_num in range(self.config.ROUNDS):
            if self.rank == 0:
                self.logger.info(f"\n=== Round {round_num + 1}/{self.config.ROUNDS} ===")
            
            avg_loss = self.federated_training_step(round_num)
            
            if self.rank == 0:
                self.train_losses.append(avg_loss)
                self.logger.info(f"Training Loss: {avg_loss:.4f}")
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if self.is_distributed:
                torch.distributed.barrier(device_ids=[self.rank])
            
            if self.rank == 0 and (round_num + 1) % (self.config.EVAL_STEPS * 2) == 0:
                val_metrics = self.evaluate(round_num)
                val_loss = val_metrics.get('val_loss', float('inf'))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(round_num, is_best=True)
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:
                    self.logger.info("Early stopping triggered")
                    break
            
            if self.rank == 0 and (round_num + 1) % (self.config.SAVE_STEPS * 2) == 0:
                self.save_checkpoint(round_num)
        
        if self.rank == 0:
            self.logger.info("\n=== Final Evaluation ===")
            final_metrics = self.evaluate()
            self.save_training_history()
            self.plot_final_results()
            self.logger.info("Memory-efficient training completed!")
            return final_metrics
        
        return {}

    def save_checkpoint(self, round_num, is_best=False):
        """Save model checkpoint"""
        if self.rank != 0:
            return
        
        checkpoint_name = f"checkpoint_round_{round_num}.pt"
        if is_best:
            checkpoint_name = "best_model.pt"
        
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, checkpoint_name)
        
        checkpoint = {
            'round': round_num,
            'server_model_state_dict': self.server_model.module.state_dict() if self.is_distributed else self.server_model.state_dict(),
            'client_model_state_dict': self.local_client_models[0].module.state_dict() if self.is_distributed and self.local_client_models else self.local_client_models[0].state_dict() if self.local_client_models else None,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def plot_final_results(self):
        """Plot final results (only on main process)"""
        if self.rank != 0:
            return
        
        try:
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
            
            if self.communication_tracker:
                comm_stats = self.communication_tracker.get_stats()
                plot_communication_costs(
                    comm_stats,
                    save_path='results/plots/communication_costs.png'
                )
            
            self.logger.info("Final plots saved to results/plots/")
        except Exception as e:
            self.logger.warning(f"Could not create plots: {e}")

    def save_training_history(self):
        """Save training history (only on main process)"""
        if self.rank != 0:
            return
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'client_losses': self.client_losses_history,
            'config': vars(self.config),
            'world_size': self.world_size
        }
        
        if self.metrics_tracker:
            history['metrics_history'] = self.metrics_tracker.history
        
        if self.communication_tracker:
            history['communication_stats'] = self.communication_tracker.get_stats()
        
        with open('results/training_history.json', 'w') as f:
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
        
        self.logger.info("Training history saved to results/training_history.json")

class SimpleTokenizer:
    """Memory-efficient simple tokenizer fallback"""
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.eos_token = '</s>'
        self.pad_token = '<pad>'
    
    def __call__(self, text, **kwargs):
        words = text.split()
        ids = [hash(word) % self.vocab_size for word in words]
        
        max_length = kwargs.get('max_length', 256)
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        if kwargs.get('padding') == 'max_length':
            while len(ids) < max_length:
                ids.append(0)
        
        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([[1] * len(ids) + [0] * (max_length - len(ids))])
        }
    
    def decode(self, ids, **kwargs):
        return ' '.join([str(id) for id in ids if id != 0])

def run_distributed_training(rank, world_size, config):
    """Run distributed training on a single GPU with memory efficiency"""
    logger = setup_logging(rank)
    try:
        setup_distributed(rank, world_size)
        trainer = MultiGPUFederatedTrainer(config, rank, world_size, is_distributed=True)
        trainer.setup_models_and_optimizers()
        final_metrics = trainer.train()
        
        if rank == 0:
            print("\n" + "="*60)
            print("MEMORY-EFFICIENT DISTRIBUTED TRAINING COMPLETED")
            print("="*60)
            print(f"Final validation loss: {final_metrics.get('val_loss', 'N/A')}")
    
    except Exception as e:
        logger.error(f"GPU {rank}: Training failed: {e}")
        raise e
    
    finally:
        cleanup_distributed()

def main():
    """Main function to start distributed training with memory management"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    config = Config()
    config.BATCH_SIZE = 1
    config.NUM_CLIENTS = 8
    config.MAX_LENGTH = 64
    
    if not torch.cuda.is_available():
        print("CUDA not available. Please use GPU for distributed training.")
        sys.exit(1)
    
    world_size = torch.cuda.device_count()
    print(f"Starting memory-efficient distributed training on {world_size} GPUs")
    print(f"Settings: batch_size={config.BATCH_SIZE}, clients={config.NUM_CLIENTS}, max_length={config.MAX_LENGTH}")
    
    # Check GPU memory and accessibility
    for i in range(world_size):
        try:
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"GPU {i}: {memory_total:.1f} GB total, {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
            # Test larger allocation
            torch.cuda.set_device(i)
            test_tensor = torch.randn(25000000, device=f'cuda:{i}')  # ~100MB
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Error accessing GPU {i}: {e}")
            sys.exit(1)
    
    # Check system resource limits
    try:
        import resource
        max_processes, _ = resource.getrlimit(resource.RLIMIT_NPROC)
        max_memory, _ = resource.getrlimit(resource.RLIMIT_AS)
        print(f"System limits: max processes={max_processes}, max memory={'unlimited' if max_memory == resource.RLIM_INFINITY else f'{max_memory/(1024**3):.2f}GB'}")
        if max_processes < 1000:
            print("Warning: Max processes limit is low. Consider increasing with 'ulimit -u 4096'")
        if max_memory != resource.RLIM_INFINITY and max_memory < 64 * (1024**3):
            print("Warning: Max memory limit is low. Consider increasing with 'ulimit -v unlimited'")
    except:
        print("Warning: Could not check system resource limits")
    
    # Aggressive cleanup of stale processes
    print("Checking for stale GPU processes...")
    gpu_processes = get_gpu_processes()
    for pid, mem in gpu_processes:
        if pid != os.getpid():
            print(f"Found process {pid} using {mem} on GPU")
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=5)
                print(f"Terminated process {pid}")
            except Exception as e:
                print(f"Failed to terminate process {pid}: {e}")
    
    # Verify cleanup
    gpu_processes = get_gpu_processes()
    if any(pid != os.getpid() for pid, _ in gpu_processes):
        print("Error: Stale processes still present on GPUs")
        sys.exit(1)
    
    # Check port availability
    port = '29500'
    for alt_port in ['29500', '29501', '29502']:
        if os.system(f"netstat -tuln | grep {alt_port} > /dev/null") == 0:
            print(f"Port {alt_port} in use.")
            continue
        port = alt_port
        break
    else:
        print("Error: No available ports (29500-29502). Please free a port.")
        sys.exit(1)
    print(f"Using port {port}...")
    os.environ['MASTER_PORT'] = port
    
    try:
        mp.spawn(
            run_distributed_training,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Distributed training failed: {e}")
        print("Falling back to single-GPU training...")
        cleanup_distributed()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        trainer = MultiGPUFederatedTrainer(config, 0, 1, is_distributed=False)
        trainer.setup_models_and_optimizers()
        final_metrics = trainer.train()
        print(f"Single-GPU training completed. Final validation loss: {final_metrics.get('val_loss', 'N/A')}")
    
    finally:
        # Final cleanup
        gpu_processes = get_gpu_processes()
        for pid, mem in gpu_processes:
            if pid != os.getpid():
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    pass

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {memory_allocated:.2f}GB / {memory_total:.2f}GB ({memory_allocated/memory_total*100:.1f}%)")

if __name__ == "__main__":
    main()
