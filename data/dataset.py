"""
MedQuad Dataset Implementation
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class MedQuadDataset(Dataset):
    """
    MedQuad dataset for medical question answering
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure required columns exist
        required_columns = ['question', 'answer']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Clean and preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Clean and preprocess the dataset"""
        # Remove rows with missing values
        self.data = self.data.dropna(subset=['question', 'answer'])
        
        # Clean text
        self.data['question'] = self.data['question'].astype(str).str.strip()
        self.data['answer'] = self.data['answer'].astype(str).str.strip()
        
        # Remove empty entries
        self.data = self.data[
            (self.data['question'].str.len() > 0) & 
            (self.data['answer'].str.len() > 0)
        ]
        
        # Create input-output pairs in conversational format
        self.data['input_text'] = "Question: " + self.data['question'] + " Answer: " + self.data['answer']
        
        print(f"Dataset preprocessed: {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Prepare input text
        input_text = row['input_text']
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'question': row['question'],
            'answer': row['answer']
        }

class MedQuadDatasetQA(Dataset):
    """
    MedQuad dataset for question-answer evaluation
    Separates questions and answers for evaluation metrics
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Clean and preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Clean and preprocess the dataset"""
        # Remove rows with missing values
        self.data = self.data.dropna(subset=['question', 'answer'])
        
        # Clean text
        self.data['question'] = self.data['question'].astype(str).str.strip()
        self.data['answer'] = self.data['answer'].astype(str).str.strip()
        
        # Remove empty entries
        self.data = self.data[
            (self.data['question'].str.len() > 0) & 
            (self.data['answer'].str.len() > 0)
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Prepare question as input
        question_text = "Question: " + row['question'] + " Answer:"
        
        # Tokenize question
        question_encoding = self.tokenizer(
            question_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,  # Leave space for answer
            return_tensors='pt'
        )
        
        # Tokenize answer for evaluation
        answer_encoding = self.tokenizer(
            row['answer'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        return {
            'question_ids': question_encoding['input_ids'].squeeze(),
            'question_mask': question_encoding['attention_mask'].squeeze(),
            'answer_ids': answer_encoding['input_ids'].squeeze(),
            'answer_mask': answer_encoding['attention_mask'].squeeze(),
            'question_text': row['question'],
            'answer_text': row['answer']
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # Handle both dataset types
    if 'input_ids' in batch[0]:
        # Regular MedQuadDataset
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'questions': [item['question'] for item in batch],
            'answers': [item['answer'] for item in batch]
        }
    else:
        # MedQuadDatasetQA
        return {
            'question_ids': torch.stack([item['question_ids'] for item in batch]),
            'question_mask': torch.stack([item['question_mask'] for item in batch]),
            'answer_ids': torch.stack([item['answer_ids'] for item in batch]),
            'answer_mask': torch.stack([item['answer_mask'] for item in batch]),
            'question_texts': [item['question_text'] for item in batch],
            'answer_texts': [item['answer_text'] for item in batch]
        }

def get_dataset_statistics(dataset):
    """Get statistics about the dataset"""
    if hasattr(dataset, 'data'):
        df = dataset.data
    else:
        # For subset datasets
        df = dataset.dataset.data.iloc[dataset.indices]
    
    stats = {
        'total_samples': len(df),
        'avg_question_length': df['question'].str.len().mean(),
        'avg_answer_length': df['answer'].str.len().mean(),
        'max_question_length': df['question'].str.len().max(),
        'max_answer_length': df['answer'].str.len().max(),
    }
    
    # Get category distribution if available
    if 'category' in df.columns:
        stats['category_distribution'] = df['category'].value_counts().to_dict()
    
    return stats
