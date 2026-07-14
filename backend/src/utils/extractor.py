import re
import urllib.parse
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from typing import Dict, Any

def clean_text(text: str) -> str:
    """
    Cleans input text by:
    - Normalizing whitespace (newlines, tabs, spaces)
    - Removing HTML tags
    - Removing URLs
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_from_url(url: str) -> Dict[str, str]:
    """
    Fetches the URL and extracts the main text and title.
    """
    try:
        # Add headers to avoid bot blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            title = soup.find('h1').text.strip()
            
        # Clean boilerplate elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        # Extract paragraph texts
        paragraphs = soup.find_all('p')
        body_text = "\n\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
        
        # If no paragraphs, try getting body text
        if not body_text:
            body_text = soup.get_text()
            
        return {
            "title": title or "Extracted URL Article",
            "content": clean_text(body_text)
        }
    except Exception as e:
        raise ValueError(f"Failed to extract text from URL: {str(e)}")

def extract_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes.
    """
    try:
        import io
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return clean_text(text)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file: {str(e)}")

def extract_from_txt(file_bytes: bytes) -> str:
    """
    Extracts text from TXT bytes.
    """
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
        return clean_text(text)
    except Exception as e:
        raise ValueError(f"Failed to parse TXT file: {str(e)}")
