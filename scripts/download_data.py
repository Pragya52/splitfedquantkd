"""
Script to download and prepare the MedQuad dataset from Kaggle
"""
import os
import sys
import pandas as pd
import requests
import zipfile
import json
from pathlib import Path

def download_medquad_dataset():
    """
    Download MedQuad dataset from Kaggle
    """
    print("Downloading MedQuad dataset from Kaggle...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        import kaggle
        print("Using Kaggle API to download MedQuad dataset...")
        
        # Download the actual MedQuad dataset
        # The MedQuad dataset is available at: kevinarvai/medquad-medical-question-answer-dataset
        kaggle.api.dataset_download_files(
            'kevinarvai/medquad-medical-question-answer-dataset', 
            path=str(data_dir), 
            unzip=True
        )
        
        # Look for the main CSV file
        possible_files = [
            "data/MedQuAD.csv",
            "data/medquad.csv",
            "data/questions_answers.csv",
            "data/medical_qa.csv"
        ]
        
        # Check for any CSV files in the downloaded data
        csv_files = list(data_dir.glob("**/*.csv"))
        
        if csv_files:
            main_csv = csv_files[0]  # Take the first CSV file found
            print(f"Found dataset file: {main_csv}")
            print(f"File size: {os.path.getsize(main_csv) / (1024*1024):.2f} MB")
            return str(main_csv)
        else:
            print("No CSV files found in downloaded dataset")
            return None
        
    except ImportError:
        print("Error: Kaggle API not available.")
        print("Please install Kaggle API:")
        print("  1. pip install kaggle")
        print("  2. Download kaggle.json from your Kaggle account")
        print("  3. Place it in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        print("  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return None
        
    except Exception as e:
        print(f"Error downloading with Kaggle API: {e}")
        print("Make sure you have:")
        print("  1. Kaggle API credentials properly configured")
        print("  2. Accepted the dataset's terms and conditions on Kaggle website")
        print("  3. Internet connection")
        return None

def load_and_format_medquad(csv_path):
    """
    Load and format the MedQuad dataset to standard format
    """
    print("Loading and formatting MedQuad dataset...")
    
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # MedQuad dataset typically has columns like 'qtype', 'Question', 'Answer', 'source'
    # We need to standardize to 'question', 'answer', 'category'
    
    # Map column names to standard format
    column_mapping = {}
    
    # Common column name variations in MedQuad dataset
    for col in df.columns:
        col_lower = col.lower()
        if 'question' in col_lower or 'q' == col_lower:
            column_mapping[col] = 'question'
        elif 'answer' in col_lower or 'a' == col_lower:
            column_mapping[col] = 'answer'
        elif 'qtype' in col_lower or 'category' in col_lower or 'source' in col_lower:
            column_mapping[col] = 'category'
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        print("Error: Could not find question and answer columns")
        print("Available columns:", list(df.columns))
        return None
    
    # If no category column, create one based on content or source
    if 'category' not in df.columns:
        if 'source' in df.columns:
            df['category'] = df['source']
        else:
            # Create categories based on question content
            df['category'] = df.apply(lambda row: categorize_medical_question(row['question'], row['answer']), axis=1)
    
    # Keep only required columns
    df = df[['question', 'answer', 'category']].copy()
    
    print(f"Formatted dataset shape: {df.shape}")
    print(f"Sample data:")
    print(df.head(2))
    
    return df

def categorize_medical_question(question, answer):
    """
    Categorize medical questions based on content
    """
    text = (str(question) + " " + str(answer)).lower()
    
    # Define medical categories with keywords
    categories = {
        'cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'blood pressure', 'hypertension', 'cholesterol', 'coronary', 'artery'],
        'diabetes': ['diabetes', 'diabetic', 'insulin', 'blood sugar', 'glucose', 'pancreas'],
        'oncology': ['cancer', 'tumor', 'oncology', 'chemotherapy', 'radiation', 'malignant', 'benign', 'carcinoma'],
        'mental_health': ['mental', 'depression', 'anxiety', 'stress', 'psychiatric', 'psychology', 'mood'],
        'respiratory': ['lung', 'respiratory', 'breathing', 'asthma', 'pneumonia', 'bronchitis', 'cough'],
        'neurological': ['brain', 'neurological', 'migraine', 'headache', 'seizure', 'stroke', 'alzheimer'],
        'dermatology': ['skin', 'dermatology', 'rash', 'acne', 'eczema', 'psoriasis'],
        'gastroenterology': ['stomach', 'digestive', 'gastro', 'intestinal', 'liver', 'colon'],
        'orthopedic': ['bone', 'joint', 'arthritis', 'fracture', 'osteoporosis', 'muscle'],
        'pediatric': ['child', 'children', 'pediatric', 'infant', 'baby', 'newborn'],
        'nutrition': ['nutrition', 'diet', 'food', 'vitamin', 'mineral', 'eating'],
        'infectious_disease': ['infection', 'virus', 'bacteria', 'antibiotic', 'vaccine', 'immunization'],
        'women_health': ['pregnancy', 'menstrual', 'gynecology', 'breast', 'ovarian'],
        'general': []  # default category
    }
    
    # Check each category
    for category, keywords in categories.items():
        if category != 'general' and any(keyword in text for keyword in keywords):
            return category
    
    return 'general'

def preprocess_dataset(df):
    """
    Preprocess the dataset for federated learning
    """
    print("Preprocessing dataset...")
    
    # Basic cleaning
    df = df.dropna(subset=['question', 'answer'])
    df['question'] = df['question'].astype(str).str.strip()
    df['answer'] = df['answer'].astype(str).str.strip()
    
    # Remove very short entries (likely incomplete)
    df = df[df['question'].str.len() >= 10]
    df = df[df['answer'].str.len() >= 20]
    
    # Remove very long entries (might be corrupted or not suitable for QA)
    df = df[df['question'].str.len() <= 1000]
    df = df[df['answer'].str.len() <= 2000]
    
    # Remove duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=['question', 'answer'])
    print(f"Removed {initial_size - len(df)} duplicate entries")
    
    # Clean text
    df['question'] = df['question'].str.replace(r'\s+', ' ', regex=True)
    df['answer'] = df['answer'].str.replace(r'\s+', ' ', regex=True)
    
    # Ensure category is string
    df['category'] = df['category'].astype(str)
    
    print(f"Preprocessed dataset size: {len(df)} samples")
    print(f"Categories distribution:")
    print(df['category'].value_counts())
    
    return df

def verify_dataset(df):
    """
    Verify the dataset is properly formatted
    """
    print("Verifying dataset...")
    
    try:
        # Check required columns
        required_columns = ['question', 'answer', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        
        # Check for empty values
        empty_questions = df['question'].isna().sum()
        empty_answers = df['answer'].isna().sum()
        
        if empty_questions > 0 or empty_answers > 0:
            print(f"Warning: Found {empty_questions} empty questions and {empty_answers} empty answers")
        
        # Check data quality
        avg_q_len = df['question'].str.len().mean()
        avg_a_len = df['answer'].str.len().mean()
        
        print(f"Dataset verification successful!")
        print(f"Total samples: {len(df)}")
        print(f"Average question length: {avg_q_len:.1f} characters")
        print(f"Average answer length: {avg_a_len:.1f} characters")
        print(f"Number of categories: {df['category'].nunique()}")
        
        # Show sample
        print("\nSample entries:")
        for i in range(min(3, len(df))):
            print(f"\nQ: {df.iloc[i]['question'][:100]}...")
            print(f"A: {df.iloc[i]['answer'][:100]}...")
            print(f"Category: {df.iloc[i]['category']}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False

def main():
    """
    Main function to download and prepare the dataset
    """
    print("MedQuad Dataset Preparation from Kaggle")
    print("=" * 45)
    
    # Check if dataset already exists
    preprocessed_path = "data/medquad_dataset_preprocessed.csv"
    if os.path.exists(preprocessed_path):
        response = input(f"Dataset already exists at {preprocessed_path}. Re-download? (y/N): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return preprocessed_path
    
    # Download dataset
    csv_path = download_medquad_dataset()
    
    if not csv_path or not os.path.exists(csv_path):
        print("Error: Could not download dataset from Kaggle")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have Kaggle API installed: pip install kaggle")
        print("2. Configure Kaggle API credentials (kaggle.json)")
        print("3. Accept dataset terms on Kaggle website")
        print("4. Check internet connection")
        sys.exit(1)
    
    # Load and format dataset
    df = load_and_format_medquad(csv_path)
    
    if df is None:
        print("Error: Could not format dataset")
        sys.exit(1)
    
    # Preprocess dataset
    df_processed = preprocess_dataset(df)
    
    # Verify dataset
    if not verify_dataset(df_processed):
        print("Error: Dataset verification failed")
        sys.exit(1)
    
    # Save preprocessed dataset
    df_processed.to_csv(preprocessed_path, index=False)
    print(f"\nPreprocessed dataset saved to: {preprocessed_path}")
    
    # Update config file if it exists
    config_path = "config.py"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Update DATA_PATH
            updated_content = content.replace(
                'DATA_PATH = "data/medquad_dataset.csv"',
                f'DATA_PATH = "{preprocessed_path}"'
            )
            
            with open(config_path, 'w') as f:
                f.write(updated_content)
            
            print(f"Updated config.py with dataset path")
            
        except Exception as e:
            print(f"Could not update config.py: {e}")
    
    print("\n" + "="*45)
    print("Dataset preparation completed successfully!")
    print(f"Dataset ready for federated learning with {len(df_processed)} samples")
    print("="*45)
    
    return preprocessed_path

if __name__ == "__main__":
    main()
