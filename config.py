"""
Configuration file for Federated Learning with MedQuad dataset
"""
import torch

class Config:
    # Dataset configuration
    DATA_PATH = "data/medquad_dataset.csv"
    MAX_LENGTH = 512
    VOCAB_SIZE = 50257
    
    # Model configuration - LLaMA 2B
    HIDDEN_DIM = 2048
    INTERMEDIATE_DIM = 5632
    NUM_LAYERS = 32
    NUM_HEADS = 32
    HEAD_DIM = 64
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 10000.0
    
    # Federated Learning configuration
    NUM_CLIENTS = 3
    ROUNDS = 100
    LOCAL_EPOCHS = 1
    CLIENT_SELECTION = "all"  # or "random"
    AGGREGATION_METHOD = "fedavg"
    
    # Training configuration
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 1000
    MAX_GRAD_NORM = 1.0
    
    # Non-IID configuration
    DIRICHLET_ALPHA = 0.5  # Lower = more non-IID
    MIN_SAMPLES_PER_CLIENT = 100
    
    # Quantization and Privacy
    SIGMA = 0.1  # Gaussian noise std
    K = 10  # Quantization sharpness
    X_MAX = 1.0
    X_MIN = -1.0
    QUANTIZATION_LEVELS = 16
    
    # Knowledge Distillation
    TEMPERATURE = 3.0
    ALPHA = 0.5  # KD loss weight
    
    # Evaluation
    EVAL_STEPS = 100
    SAVE_STEPS = 500
    LOGGING_STEPS = 50
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIXED_PRECISION = True
    
    # Experiment tracking
    USE_WANDB = False
    PROJECT_NAME = "federated-medquad-llama"
    EXPERIMENT_NAME = "baseline"
    
    # Metrics to track
    METRICS = [
        'accuracy',
        'f1_score',
        'bleu_score',
        'rouge_l',
        'perplexity',
        'exact_match'
    ]
