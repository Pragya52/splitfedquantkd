"""
Evaluation metrics for medical question answering
"""
import torch
import numpy as np
from collections import Counter
import re
from typing import List, Dict, Any
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    """
    Preprocess text for evaluation
    
    Args:
        text: Input text string
    
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation for some metrics
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    
    return text, text_no_punct

def exact_match_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match score
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        Exact match score (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    exact_matches = 0
    for pred, ref in zip(predictions, references):
        pred_clean, _ = preprocess_text(pred)
        ref_clean, _ = preprocess_text(ref)
        
        if pred_clean == ref_clean:
            exact_matches += 1
    
    return exact_matches / len(predictions)

def f1_score_token_level(predictions: List[str], references: List[str]) -> float:
    """
    Compute token-level F1 score
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        Average F1 score
    """
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        _, pred_tokens = preprocess_text(pred)
        _, ref_tokens = preprocess_text(ref)
        
        pred_tokens = set(pred_tokens.split())
        ref_tokens = set(ref_tokens.split())
        
        if len(ref_tokens) == 0:
            f1_scores.append(1.0 if len(pred_tokens) == 0 else 0.0)
            continue
        
        if len(pred_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        # Calculate precision, recall, and F1
        common_tokens = pred_tokens.intersection(ref_tokens)
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

def bleu_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU scores (BLEU-1 to BLEU-4)
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        Dictionary with BLEU scores
    """
    smoothing = SmoothingFunction().method1
    bleu_scores = {f'bleu_{i}': [] for i in range(1, 5)}
    
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = [nltk.word_tokenize(ref.lower())]
        
        for i in range(1, 5):
            weights = [1/i] * i + [0] * (4-i)
            try:
                score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothing)
                bleu_scores[f'bleu_{i}'].append(score)
            except ZeroDivisionError:
                bleu_scores[f'bleu_{i}'].append(0.0)
    
    # Calculate averages
    for key in bleu_scores:
        bleu_scores[key] = np.mean(bleu_scores[key])
    
    return bleu_scores

def rouge_score_eval(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        Dictionary with ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # Calculate averages
    for key in rouge_scores:
        rouge_scores[key] = np.mean(rouge_scores[key])
    
    return rouge_scores

def perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute perplexity from logits and labels
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in loss calculation
    
    Returns:
        Perplexity score
    """
    # Flatten logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    loss = loss_fct(shift_logits, shift_labels)
    
    # Convert to perplexity
    perplexity_score = torch.exp(loss).item()
    
    return perplexity_score

def semantic_similarity_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute semantic similarity using simple word overlap
    This is a simplified version - in practice, you might use sentence embeddings
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        Average semantic similarity score
    """
    similarities = []
    
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        if len(ref_words) == 0:
            similarities.append(1.0 if len(pred_words) == 0 else 0.0)
            continue
        
        # Jaccard similarity
        intersection = len(pred_words.intersection(ref_words))
        union = len(pred_words.union(ref_words))
        
        similarity = intersection / union if union > 0 else 0.0
        similarities.append(similarity)
    
    return np.mean(similarities)

def calculate_all_metrics(predictions: List[str], references: List[str], 
                         logits: torch.Tensor = None, labels: torch.Tensor = None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        logits: Model logits (optional, for perplexity)
        labels: Target labels (optional, for perplexity)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Text-based metrics
    metrics['exact_match'] = exact_match_score(predictions, references)
    metrics['f1_token'] = f1_score_token_level(predictions, references)
    metrics['semantic_similarity'] = semantic_similarity_score(predictions, references)
    
    # BLEU scores
    bleu_scores = bleu_score(predictions, references)
    metrics.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = rouge_score_eval(predictions, references)
    metrics.update(rouge_scores)
    
    # Perplexity (if logits and labels are provided)
    if logits is not None and labels is not None:
        metrics['perplexity'] = perplexity(logits, labels)
    
    return metrics

def medical_specific_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate medical domain-specific metrics
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
    
    Returns:
        Dictionary with medical-specific metrics
    """
    metrics = {}
    
    # Medical terms extraction (simplified)
    medical_terms = [
        'diagnosis', 'treatment', 'symptom', 'disease', 'medication', 'therapy',
        'patient', 'doctor', 'hospital', 'clinic', 'medical', 'health',
        'condition', 'syndrome', 'disorder', 'infection', 'virus', 'bacteria',
        'cancer', 'tumor', 'diabetes', 'hypertension', 'cardiovascular'
    ]
    
    # Medical term coverage
    med_term_coverage = []
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        ref_med_terms = set(term for term in medical_terms if term in ref_lower)
        pred_med_terms = set(term for term in medical_terms if term in pred_lower)
        
        if len(ref_med_terms) == 0:
            coverage = 1.0 if len(pred_med_terms) == 0 else 0.0
        else:
            coverage = len(pred_med_terms.intersection(ref_med_terms)) / len(ref_med_terms)
        
        med_term_coverage.append(coverage)
    
    metrics['medical_term_coverage'] = np.mean(med_term_coverage)
    
    # Answer length similarity
    length_similarities = []
    for pred, ref in zip(predictions, references):
        pred_len = len(pred.split())
        ref_len = len(ref.split())
        
        if ref_len == 0:
            similarity = 1.0 if pred_len == 0 else 0.0
        else:
            similarity = 1.0 - abs(pred_len - ref_len) / max(pred_len, ref_len, 1)
        
        length_similarities.append(max(0.0, similarity))
    
    metrics['length_similarity'] = np.mean(length_similarities)
    
    return metrics

def print_metrics_summary(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    # Group metrics by category
    categories = {
        'Accuracy Metrics': ['exact_match', 'f1_token', 'semantic_similarity'],
        'BLEU Scores': ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4'],
        'ROUGE Scores': ['rouge1', 'rouge2', 'rougeL'],
        'Language Model Metrics': ['perplexity'],
        'Medical-Specific': ['medical_term_coverage', 'length_similarity']
    }
    
    for category, metric_names in categories.items():
        category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
        if category_metrics:
            print(f"\n{category}:")
            print("-" * 30)
            for metric_name, value in category_metrics.items():
                if metric_name == 'perplexity':
                    print(f"  {metric_name:20}: {value:.4f}")
                else:
                    print(f"  {metric_name:20}: {value:.4f} ({value*100:.2f}%)")
    
    # Print any remaining metrics
    all_categorized = set()
    for metric_list in categories.values():
        all_categorized.update(metric_list)
    
    remaining_metrics = {k: v for k, v in metrics.items() if k not in all_categorized}
    if remaining_metrics:
        print(f"\nOther Metrics:")
        print("-" * 30)
        for metric_name, value in remaining_metrics.items():
            print(f"  {metric_name:20}: {value:.4f}")
    
    print(f"{'='*50}")

class MetricsTracker:
    """
    Track metrics over training rounds
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.history = {}
    
    def update(self, metrics: Dict[str, float], round_num: int = None):
        """
        Update metrics for a given round
        
        Args:
            metrics: Dictionary of metrics
            round_num: Round number (if None, appends to existing history)
        """
        for metric_name, value in metrics.items():
            if metric_name not in self.history:
                self.history[metric_name] = []
            
            if round_num is not None:
                # Ensure history is long enough
                while len(self.history[metric_name]) <= round_num:
                    self.history[metric_name].append(None)
                self.history[metric_name][round_num] = value
            else:
                self.history[metric_name].append(value)
    
    def get_best_scores(self) -> Dict[str, float]:
        """
        Get best scores for each metric
        
        Returns:
            Dictionary with best scores
        """
        best_scores = {}
        for metric_name, values in self.history.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                if metric_name == 'perplexity':
                    best_scores[metric_name] = min(valid_values)
                else:
                    best_scores[metric_name] = max(valid_values)
        
        return best_scores
    
    def get_latest_scores(self) -> Dict[str, float]:
        """
        Get latest scores for each metric
        
        Returns:
            Dictionary with latest scores
        """
        latest_scores = {}
        for metric_name, values in self.history.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                latest_scores[metric_name] = valid_values[-1]
        
        return latest_scores
