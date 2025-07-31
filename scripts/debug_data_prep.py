"""
Debug script to understand your dataset structure
"""
import pandas as pd
import os

def debug_dataset():
    """Debug and prepare dataset step by step"""
    
    # Find your dataset
    data_paths = [
        "data/MedQuAD.csv",
        "data/medquad.csv", 
        "data/medical_qa.csv"
    ]
    
    csv_path = None
    for path in data_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if not csv_path:
        csv_path = input("Enter path to your CSV file: ").strip().strip('"').strip("'")
    
    print(f"Loading: {csv_path}")
    df_original = pd.read_csv(csv_path)
    
    print("="*50)
    print("ORIGINAL DATASET INFO:")
    print(f"Shape: {df_original.shape}")
    print(f"Columns: {list(df_original.columns)}")
    print(f"Data types:\n{df_original.dtypes}")
    print("\nFirst few rows:")
    print(df_original.head())
    print("="*50)
    
    # Let's manually map columns
    print("\nCOLUMN MAPPING:")
    columns = list(df_original.columns)
    
    # Auto-detect likely columns
    question_col = None
    answer_col = None
    category_col = None
    
    for col in columns:
        col_lower = col.lower()
        if 'question' in col_lower and question_col is None:
            question_col = col
            print(f"Found question column: {col}")
        elif 'answer' in col_lower and answer_col is None:
            answer_col = col
            print(f"Found answer column: {col}")
        elif any(word in col_lower for word in ['category', 'type', 'source', 'qtype']) and category_col is None:
            category_col = col
            print(f"Found category column: {col}")
    
    # Manual selection if auto-detection failed
    if not question_col:
        print("\nAvailable columns:")
        for i, col in enumerate(columns):
            print(f"  {i}: {col}")
        idx = int(input("Select QUESTION column number: "))
        question_col = columns[idx]
    
    if not answer_col:
        print("\nAvailable columns:")
        for i, col in enumerate(columns):
            print(f"  {i}: {col}")
        idx = int(input("Select ANSWER column number: "))
        answer_col = columns[idx]
        
    if not category_col:
        print("\nAvailable columns:")
        for i, col in enumerate(columns):
            print(f"  {i}: {col}")
        response = input("Select CATEGORY column number (or press Enter to auto-create): ")
        if response.strip():
            category_col = columns[int(response)]
    
    print(f"\nUsing mapping:")
    print(f"  Question: {question_col}")
    print(f"  Answer: {answer_col}")
    print(f"  Category: {category_col}")
    
    # Create new dataframe step by step
    print("\nCREATING NEW DATAFRAME:")
    
    # Step 1: Extract question column
    print(f"Extracting questions from '{question_col}'...")
    questions = df_original[question_col].copy()
    print(f"Questions type: {type(questions)}")
    print(f"Questions shape: {questions.shape}")
    print(f"Sample question: {questions.iloc[0]}")
    
    # Step 2: Extract answer column
    print(f"\nExtracting answers from '{answer_col}'...")
    answers = df_original[answer_col].copy()
    print(f"Answers type: {type(answers)}")
    print(f"Answers shape: {answers.shape}")
    print(f"Sample answer: {str(answers.iloc[0])[:100]}...")
    
    # Step 3: Extract or create category
    if category_col:
        print(f"\nExtracting categories from '{category_col}'...")
        categories = df_original[category_col].copy()
    else:
        print("\nCreating categories...")
        categories = pd.Series(['general'] * len(df_original))
    
    print(f"Categories type: {type(categories)}")
    print(f"Categories shape: {categories.shape}")
    print(f"Unique categories: {categories.value_counts().head()}")
    
    # Step 4: Create final dataframe
    print("\nCREATING FINAL DATAFRAME:")
    final_df = pd.DataFrame({
        'question': questions,
        'answer': answers, 
        'category': categories
    })
    
    print(f"Final df shape: {final_df.shape}")
    print(f"Final df columns: {list(final_df.columns)}")
    print(f"Final df dtypes:\n{final_df.dtypes}")
    
    # Step 5: Simple cleaning
    print("\nCLEANING DATA:")
    
    # Convert to string safely
    print("Converting to strings...")
    final_df['question'] = final_df['question'].astype(str)
    final_df['answer'] = final_df['answer'].astype(str)
    final_df['category'] = final_df['category'].astype(str)
    
    # Strip whitespace
    print("Stripping whitespace...")
    final_df['question'] = final_df['question'].str.strip()
    final_df['answer'] = final_df['answer'].str.strip()
    final_df['category'] = final_df['category'].str.strip()
    
    # Remove empty
    print("Removing empty entries...")
    initial_len = len(final_df)
    final_df = final_df[(final_df['question'].str.len() > 5) & 
                        (final_df['answer'].str.len() > 10)]
    print(f"Removed {initial_len - len(final_df)} empty entries")
    
    # Remove duplicates
    print("Removing duplicates...")
    initial_len = len(final_df)
    final_df = final_df.drop_duplicates(subset=['question', 'answer'])
    print(f"Removed {initial_len - len(final_df)} duplicates")
    
    print(f"\nFINAL DATASET:")
    print(f"Shape: {final_df.shape}")
    print(f"Categories: {final_df['category'].value_counts().head()}")
    
    # Show samples
    print(f"\nSAMPLE DATA:")
    for i in range(min(3, len(final_df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Q: {final_df.iloc[i]['question'][:100]}...")
        print(f"A: {final_df.iloc[i]['answer'][:100]}...")
        print(f"C: {final_df.iloc[i]['category']}")
    
    # Save
    output_path = "data/medquad_dataset_preprocessed.csv"
    os.makedirs("data", exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    debug_dataset()
