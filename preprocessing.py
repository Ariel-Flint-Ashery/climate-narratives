#%%
import re
from datetime import datetime
import os
import pandas as pd
#%%
def load_text(filepath):
    """Load text from a .txt file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_date(text):
    """
    Extract and convert date from text to datetime object
    
    Args:
        text (str): Input text containing date information
        
    Returns:
        datetime: The extracted date as a datetime object
        None: If no valid date is found
    """
    # Try different date patterns
    date_patterns = [
        # Pattern for "March 11, 2011 Friday"
        (r'([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})', '%B %d, %Y'),
        # Pattern for "11 March 2011"
        (r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', '%d %B %Y'),
        # Pattern for "2011-03-11"
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d')
    ]
    
    for pattern, date_format in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                if date_format == '%B %d, %Y':
                    date_str = f"{match.group(1)} {match.group(2)}, {match.group(3)}"
                elif date_format == '%d %B %Y':
                    date_str = f"{match.group(1)} {match.group(2)} {match.group(3)}"
                else:
                    date_str = match.group(0)
                    
                return datetime.strptime(date_str, date_format)
            except ValueError:
                continue
    return None

def extract_length(text):
    """
    Extract the length value from a string containing "Length: X words"
    
    Args:
        text (str): Input text containing length information
        
    Returns:
        int: The extracted length value
        None: If no length value is found
    """
    pattern = r'Length:\s*(\d+)\s*words'
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))
    return None

def extract_author(text):
    """
    Extract author information from text containing a byline
    
    Args:
        text (str): Input text containing byline information
        
    Returns:
        str: The extracted author(s) or "unknown" if no byline is found
    """
    # Look for byline pattern, considering various formats and possible line endings
    pattern = r'Byline:\s*([^,\n]*(?:,\s*[^,\n]+)*)'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # Clean up the author string by removing extra whitespace
        author = match.group(1).strip()
        # Remove trailing comma if it exists
        author = author.rstrip(',')
        return author
    return None

def extract_title(filename):
    """
    Extract title from filename by removing extension and converting underscores to spaces
    
    Args:
        filename (str): Input filename (e.g., "some_title.txt")
        
    Returns:
        str: The extracted title with spaces instead of underscores
    """
    # Remove .txt extension and convert underscores to spaces
    title = os.path.splitext(filename)[0]  # removes .txt
    return title.strip()

def extract_metadata(text, filename):
    """
    Extract date, length, and author metadata from text
    
    Args:
        text (str): Input text containing metadata information
        
    Returns:
        tuple: (datetime object, integer length, string author)
    """
    date = extract_date(text)
    length = extract_length(text)
    author = extract_author(text)
    title = extract_title(filename)
    return date, length, author, title

def load_or_create_df(filepath):
    """
    Load an existing news DataFrame from CSV or create a new one if it doesn't exist.
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded or newly created DataFrame
    """
    if os.path.exists(filepath):
        try:
            # Read existing CSV
            df = pd.read_csv(filepath)
            
            # Convert string representations of lists back to actual lists
            # This is needed because CSV storage converts lists to strings
            # if 'contrarian_claims' in df.columns:
            #     df['contrarian_claims'] = df['contrarian_claims'].apply(eval)
            # if 'frame_summaries' in df.columns:
            #     df['frame_summaries'] = df['frame_summaries'].apply(eval)
                
            print(f"Loaded existing DataFrame from {filepath}")
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Creating new DataFrame instead")
    
    # Create new DataFrame if file doesn't exist or there was an error
    columns = ['Title', 'Author', 'Source', 'Year', 'Month', 
              'contrarian_claims', 'contrarian_claim_ids', 'frame_summaries']
    return pd.DataFrame(columns=columns)

def add_article_to_df(df, article_data):
    """
    Add a new article to the DataFrame using a dictionary.
    
    Parameters:
    df (pandas.DataFrame): Existing news articles DataFrame
    article_data (dict): Dictionary containing article information
    
    Returns:
    pandas.DataFrame: Updated DataFrame with new article
    """
    return pd.concat([df, pd.DataFrame([article_data])], ignore_index=True)

def save_df(df, filepath):
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to save
    filepath (str): Path where to save the CSV file
    """
    try:
        df.to_csv(filepath, index=False)
        print(f"Successfully saved DataFrame to {filepath}")
    except Exception as e:
        print(f"Error saving CSV: {e}")