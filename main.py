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
                collate_
