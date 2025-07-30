"""
Non-IID Data Splitting for Federated Learning
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def create_non_iid_split(dataset, num_clients, alpha=0.5, min_samples_per_client=100):
    """
    Create non-IID data splits using Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        min_samples_per_client: Minimum samples per client
    
    Returns:
        List of client datasets
    """
    print(f"Creating non-IID split with alpha={alpha} for {num_clients} clients")
    
    # Get labels or create pseudo-labels based on text characteristics
    if hasattr(dataset, 'data'):
        df = dataset.data
    else:
        # Handle subset datasets
        df = dataset.dataset.data.iloc[dataset.indices] if hasattr(dataset, 'indices') else dataset.dataset.data
    
    # Create labels based on question characteristics
    labels = create_pseudo_labels(df)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    print(f"Number of classes: {num_classes}")
    
    # Create label to indices mapping
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    # Sample proportions for each client using Dirichlet distribution
    client_proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Distribute samples to clients
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx, label in enumerate(unique_labels):
        class_indices = label_to_indices[label]
        np.random.shuffle(class_indices)
        
        # Calculate number of samples for each client
        proportions = client_proportions[class_idx]
        client_counts = (proportions * len(class_indices)).astype(int)
        
        # Ensure we distribute all samples
        client_counts[-1] += len(class_indices) - client_counts.sum()
        
        # Distribute indices
        start_idx = 0
        for client_idx in range(num_clients):
            end_idx = start_idx + client_counts[client_idx]
            client_indices[client_idx].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Ensure minimum samples per client
    for client_idx in range(num_clients):
        if len(client_indices[client_idx]) < min_samples_per_client:
            # Randomly sample from other clients
            other_clients = [i for i in range(num_clients) if i != client_idx and len(client_indices[i]) > min_samples_per_client]
            needed = min_samples_per_client - len(client_indices[client_idx])
            
            for _ in range(needed):
                if other_clients:
                    donor_client = np.random.choice(other_clients)
                    if len(client_indices[donor_client]) > min_samples_per_client:
                        donated_idx = client_indices[donor_client].pop()
                        client_indices[client_idx].append(donated_idx)
    
    # Create client datasets
    client_datasets = []
    for client_idx in range(num_clients):
        if hasattr(dataset, 'indices'):
            # Handle subset datasets
            actual_indices = [dataset.indices[i] for i in client_indices[client_idx]]
            client_dataset = Subset(dataset.dataset, actual_indices)
        else:
            client_dataset = Subset(dataset, client_indices[client_idx])
        client_datasets.append(client_dataset)
    
    # Print statistics
    print("Client data distribution:")
    for i, client_dataset in enumerate(client_datasets):
        print(f"Client {i+1}: {len(client_dataset)} samples")
    
    # Visualize distribution
    visualize_non_iid_distribution(client_datasets, labels, unique_labels, num_clients)
    
    return client_datasets

def create_pseudo_labels(df):
    """
    Create pseudo-labels based on text characteristics
    """
    labels = []
    
    for _, row in df.iterrows():
        question = str(row['question']).lower()
        answer = str(row['answer']).lower()
        
        # Create labels based on medical categories/topics
        if any(keyword in question for keyword in ['heart', 'cardiac', 'cardiovascular']):
            label = 0  # Cardiovascular
        elif any(keyword in question for keyword in ['diabetes', 'blood sugar', 'insulin']):
            label = 1  # Diabetes
        elif any(keyword in question for keyword in ['cancer', 'tumor', 'oncology']):
            label = 2  # Oncology
        elif any(keyword in question for keyword in ['mental', 'depression', 'anxiety', 'psychiatric']):
            label = 3  # Mental Health
        elif any(keyword in question for keyword in ['child', 'pediatric', 'infant', 'baby']):
            label = 4  # Pediatrics
        elif any(keyword in question for keyword in ['drug', 'medication', 'treatment', 'therapy']):
            label = 5  # Treatment/Medication
        elif any(keyword in question for keyword in ['symptom', 'diagnosis', 'condition']):
            label = 6  # Symptoms/Diagnosis
        elif any(keyword in question for keyword in ['nutrition', 'diet', 'food', 'eating']):
            label = 7  # Nutrition
        elif any(keyword in question for keyword in ['exercise', 'fitness', 'physical']):
            label = 8  # Physical Activity
        else:
            # Hash-based assignment for remaining questions
            label = 9 + (hash(question) % 6)  # Labels 9-14 for other categories
        
        labels.append(label)
    
    return np.array(labels)

def visualize_non_iid_distribution(client_datasets, all_labels, unique_labels, num_clients):
    """
    Visualize the non-IID distribution across clients
    """
    # Create distribution matrix
    client_label_counts = np.zeros((num_clients, len(unique_labels)))
    
    for client_idx, client_dataset in enumerate(client_datasets):
        if hasattr(client_dataset, 'indices'):
            client_labels = all_labels[client_dataset.indices]
        else:
            client_labels = all_labels[list(range(len(client_dataset)))]
        
        for label in unique_labels:
            client_label_counts[client_idx, label] = np.sum(client_labels == label)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        client_label_counts, 
        annot=True, 
        fmt='g',
        xticklabels=[f'Class {i}' for i in unique_labels],
        yticklabels=[f'Client {i+1}' for i in range(num_clients)],
        cmap='Blues'
    )
    plt.title('Non-IID Data Distribution Across Clients')
    plt.xlabel('Classes')
    plt.ylabel('Clients')
    plt.tight_layout()
    plt.savefig('results/plots/non_iid_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar plot for each client
    fig, axes = plt.subplots(1, num_clients, figsize=(4*num_clients, 6))
    if num_clients == 1:
        axes = [axes]
    
    for client_idx in range(num_clients):
        axes[client_idx].bar(unique_labels, client_label_counts[client_idx])
        axes[client_idx].set_title(f'Client {client_idx+1}')
        axes[client_idx].set_xlabel('Classes')
        axes[client_idx].set_ylabel('Number of Samples')
        axes[client_idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/plots/client_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Distribution plots saved to results/plots/")

def calculate_non_iid_metrics(client_datasets, all_labels):
    """
    Calculate metrics to quantify the degree of non-IID-ness
    """
    num_clients = len(client_datasets)
    unique_labels = np.unique(all_labels)
    
    # Calculate client label distributions
    client_distributions = []
    for client_dataset in client_datasets:
        if hasattr(client_dataset, 'indices'):
            client_labels = all_labels[client_dataset.indices]
        else:
            client_labels = all_labels[list(range(len(client_dataset)))]
        
        # Calculate distribution
        distribution = np.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            distribution[i] = np.sum(client_labels == label) / len(client_labels)
        client_distributions.append(distribution)
    
    client_distributions = np.array(client_distributions)
    
    # Calculate global distribution
    global_distribution = np.zeros(len(unique_labels))
    total_samples = sum(len(ds) for ds in client_datasets)
    for i, label in enumerate(unique_labels):
        global_distribution[i] = np.sum(all_labels == label) / len(all_labels)
    
    # Calculate metrics
    metrics = {}
    
    # 1. KL Divergence from global distribution
    kl_divergences = []
    for client_dist in client_distributions:
        # Add small epsilon to avoid log(0)
        client_dist = client_dist + 1e-10
        global_dist = global_distribution + 1e-10
        kl_div = np.sum(client_dist * np.log(client_dist / global_dist))
        kl_divergences.append(kl_div)
    
    metrics['avg_kl_divergence'] = np.mean(kl_divergences)
    metrics['max_kl_divergence'] = np.max(kl_divergences)
    
    # 2. Earth Mover's Distance (Wasserstein-1)
    try:
        from scipy.stats import wasserstein_distance
        emd_distances = []
        for client_dist in client_distributions:
            emd = wasserstein_distance(client_dist, global_distribution)
            emd_distances.append(emd)
        
        metrics['avg_emd'] = np.mean(emd_distances)
        metrics['max_emd'] = np.max(emd_distances)
    except ImportError:
        print("scipy not available for EMD calculation")
    
    # 3. Cosine similarity
    cosine_similarities = []
    for client_dist in client_distributions:
        cosine_sim = np.dot(client_dist, global_distribution) / (
            np.linalg.norm(client_dist) * np.linalg.norm(global_distribution)
        )
        cosine_similarities.append(cosine_sim)
    
    metrics['avg_cosine_similarity'] = np.mean(cosine_similarities)
    metrics['min_cosine_similarity'] = np.min(cosine_similarities)
    
    # 4. Number of missing classes per client
    missing_classes = []
    for client_dist in client_distributions:
        missing = np.sum(client_dist == 0)
        missing_classes.append(missing)
    
    metrics['avg_missing_classes'] = np.mean(missing_classes)
    metrics['max_missing_classes'] = np.max(missing_classes)
    
    return metrics
