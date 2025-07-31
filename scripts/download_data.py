"""
Script to prepare the MedQuad dataset from local file
"""
import os
import sys
import pandas as pd
import json
from pathlib import Path

def find_local_dataset():
    """
    Find MedQuad dataset in local directories
    """
    print("Looking for MedQuad dataset in local directories...")
    
    # Common locations to check
    search_paths = [
        "data/",
        "./",
        "../data/",
        "datasets/",
        "../datasets/"
    ]
    
    # Common file names for MedQuad dataset
    possible_filenames = [
        "MedQuAD.csv",
        "medquad.csv",
        "medquad_dataset.csv",
        "medical_qa.csv",
        "questions_answers.csv",
        "qa_dataset.csv"
    ]
    
    # Search for dataset files
    found_files = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            for filename in possible_filenames:
                file_path = os.path.join(search_path, filename)
                if os.path.exists(file_path):
                    found_files.append(file_path)
    
    # Also check for any CSV files in data directory
    data_dir = Path("data")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        found_files.extend([str(f) for f in csv_files])
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if found_files:
        print(f"Found {len(found_files)} potential dataset file(s):")
        for i, file_path in enumerate(found_files):
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  {i+1}. {file_path} ({file_size:.2f} MB)")
        
        # If multiple files found, let user choose
        if len(found_files) > 1:
            while True:
                try:
                    choice = input(f"\nSelect dataset file (1-{len(found_files)}): ")
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(found_files):
                        return found_files[choice_idx]
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            return found_files[0]
    
    return None

def get_dataset_path():
    """
    Get dataset path from user or find automatically
    """
    print("MedQuad Dataset Preparation from Local File")
    print("=" * 45)
    
    # First, try to find automatically
    auto_found = find_local_dataset()
    
    if auto_found:
        use_auto = input(f"\nUse automatically found file '{auto_found}'? (Y/n): ")
        if use_auto.lower() != 'n':
            return auto_found
    
    # Manual path input
    print("\nPlease provide the path to your MedQuad dataset CSV file:")
    print("Examples:")
    print("  - data/MedQuAD.csv")
    print("  - /path/to/your/dataset.csv")
    print("  - C:\\Users\\username\\Documents\\dataset.csv")
    
    while True:
        file_path = input("\nEnter dataset file path: ").strip().strip('"').strip("'")
        
        if not file_path:
            print("Please enter a valid file path.")
            continue
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"Found file: {file_path} ({file_size:.2f} MB)")
            return file_path
        else:
            print(f"File not found: {file_path}")
            retry = input("Try again? (Y/n): ")
            if retry.lower() == 'n':
                return None

def load_and_format_medquad(csv_path):
    """
    Load and format the MedQuad dataset to standard format
    """
    print("Loading and formatting MedQuad dataset...")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("Error: Could not read CSV file with any encoding")
            return None
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # MedQuad dataset typically has columns like 'qtype', 'Question', 'Answer', 'source'
    # We need to standardize to 'question', 'answer', 'category'
    
    # Map column names to standard format
    column_mapping = {}
    
    # Common column name variations in MedQuad dataset
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(keyword in col_lower for keyword in ['question', 'query', 'q']):
            column_mapping[col] = 'question'
        elif any(keyword in col_lower for keyword in ['answer', 'response', 'a']):
            column_mapping[col] = 'answer'
        elif any(keyword in col_lower for keyword in ['qtype', 'category', 'source', 'type', 'class']):
            column_mapping[col] = 'category'
    
    print(f"Column mapping: {column_mapping}")
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        print("Error: Could not find question and answer columns")
        print("Available columns:", list(df.columns))
        print("\nPlease make sure your CSV has columns containing questions and answers.")
        print("Common column names: 'Question', 'Answer', 'query', 'response', etc.")
        return None
    
    # If no category column, create one based on content or use existing columns
    if 'category' not in df.columns:
        # Check if there are other useful columns for categories
        remaining_cols = [col for col in df.columns if col not in ['question', 'answer']]
        if remaining_cols:
            print(f"Using '{remaining_cols[0]}' as category column")
            df['category'] = df[remaining_cols[0]].astype(str)
        else:
            # Create categories based on question content
            print("Creating categories based on medical content...")
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
    initial_size = len(df)
    df = df[df['question'].str.len() >= 10]
    df = df[df['answer'].str.len() >= 20]
    print(f"Removed {initial_size - len(df)} entries with very short questions or answers")
    
    # Remove very long entries (might be corrupted or not suitable for QA)
    initial_size = len(df)
    df = df[df['question'].str.len() <= 1000]
    df = df[df['answer'].str.len() <= 2000]
    print(f"Removed {initial_size - len(df)} entries with very long text")
    
    # Remove duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=['question', 'answer'])
    print(f"Removed {initial_size - len(df)} duplicate entries")
    
    # Clean text - normalize whitespace
    df['question'] = df['question'].str.replace(r'\s+', ' ', regex=True)
    df['answer'] = df['answer'].str.replace(r'\s+', ' ', regex=True)
    
    # Ensure category is string and clean it
    df['category'] = df['category'].astype(str).str.strip().str.lower()
    
    print(f"Preprocessed dataset size: {len(df)} samples")
    print(f"Categories distribution:")
    category_counts = df['category'].value_counts()
    print(category_counts)
    
    return df

def verify_dataset(df):
    """
    Verify the dataset is properly formatted
    """
    print("\nVerifying dataset...")
    
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
        
        # Show sample entries
        print("\nSample entries:")
        for i in range(min(3, len(df))):
            print(f"\n--- Sample {i+1} ---")
            print(f"Q: {df.iloc[i]['question'][:150]}...")
            print(f"A: {df.iloc[i]['answer'][:150]}...")
            print(f"Category: {df.iloc[i]['category']}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False

def main():
    """
    Main function to prepare the dataset from local file
    """
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if preprocessed dataset already exists
    preprocessed_path = "data/medquad_dataset_preprocessed.csv"
    if os.path.exists(preprocessed_path):
        response = input(f"Preprocessed dataset already exists at {preprocessed_path}. Re-process? (y/N): ")
        if response.lower() != 'y':
            print("Using existing preprocessed dataset.")
            return preprocessed_path
    
    # Get dataset path
    csv_path = get_dataset_path()
    
    if not csv_path:
        print("Error: No dataset file provided")
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
