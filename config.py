"""
Configuration file for Federated Learning with MedQuad dataset
Original LLaMA 2B Model Specifications
"""
import torch

class Config:
    # Dataset configuration
    DATA_PATH = "data/medquad_dataset_preprocessed.csv"
    MAX_LENGTH = 2048  # LLaMA 2B original context length
    VOCAB_SIZE = 50257  # Match DialoGPT tokenizer (or use 32000 for LLaMA tokenizer)
    
    # Model configuration - Original LLaMA 2B
    HIDDEN_DIM = 2048
    INTERMEDIATE_DIM = 5632  # FFN dimension (2.75 * hidden_dim)
    NUM_LAYERS = 24         # LLaMA 2B has 24 layers, not 32
    NUM_HEADS = 32          # 32 attention heads
    HEAD_DIM = 64           # 2048 / 32 = 64
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 10000.0
    
    # Federated Learning configuration
    NUM_CLIENTS = 3
    ROUNDS = 100
    LOCAL_EPOCHS = 1
    CLIENT_SELECTION = "all"  # or "random"
    AGGREGATION_METHOD = "fedavg"
    
    # Training configuration - GPU optimized for 2B model
    BATCH_SIZE = 2          # Reduced for memory efficiency
    LEARNING_RATE = 3e-4    # LLaMA original learning rate
    WEIGHT_DECAY = 0.1      # LLaMA original weight decay
    WARMUP_STEPS = 2000     # More warmup for larger model
    MAX_GRAD_NORM = 1.0
    
    # Memory optimizations for large model
    GRADIENT_CHECKPOINTING = True
    MIXED_PRECISION = True
    
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
    
    # Evaluation - More frequent for large model monitoring
    EVAL_STEPS = 50    # More frequent evaluation
    SAVE_STEPS = 100   # More frequent saves
    LOGGING_STEPS = 10 # More frequent logging
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    # Device - GPU required for 2B model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIXED_PRECISION = True
    
    # Memory management for large model
    PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512,expandable_segments:True"
    
    # Experiment tracking
    USE_WANDB = False
    PROJECT_NAME = "federated-medquad-llama2b"
    EXPERIMENT_NAME = "llama2b_baseline"
    
    # Metrics to track
    METRICS = [
        'accuracy',
        'f1_score',
        'bleu_score',
        'rouge_l',
        'perplexity',
        'exact_match'
    ]
    
    # LLaMA 2B specific optimizations
    USE_ROPE_SCALING = False    # Standard RoPE
    USE_FLASH_ATTENTION = True  # If available
    ATTENTION_DROPOUT = 0.0     # LLaMA doesn't use attention dropout
    RESIDUAL_DROPOUT = 0.0      # LLaMA doesn't use residual dropout
    
    # Training schedule for large model
    SCHEDULER_TYPE = "cosine"   # Cosine annealing
    MIN_LR_RATIO = 0.1         # Minimum learning rate ratio
    
    # Model initialization
    INITIALIZER_RANGE = 0.02   # Standard deviation for weight initialization
