#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_from_markdown(file_path):
    """Extract text content from markdown files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.info(f"Successfully extracted text from markdown file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error extracting text from markdown file {file_path}: {e}")
        return ""

def extract_from_pdf(file_path):
    """Extract text content from PDF files."""
    try:
        # Try to import PyPDF2 first
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except ImportError:
            # Fall back to pdfplumber if PyPDF2 is not available
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or "" + "\n"
        
        logger.info(f"Successfully extracted text from PDF file: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF file {file_path}: {e}")
        return ""

def process_files(input_dir, output_file):
    """Process all PDF and Markdown files in the input directory and write to output file."""
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist.")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Find all PDF and Markdown files
    pdf_files = glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True)
    md_files = glob.glob(os.path.join(input_dir, "**", "*.md"), recursive=True)
    markdown_files = glob.glob(os.path.join(input_dir, "**", "*.markdown"), recursive=True)
    
    all_files = pdf_files + md_files + markdown_files
    
    if not all_files:
        logger.warning(f"No PDF or Markdown files found in {input_dir}")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files and {len(md_files) + len(markdown_files)} Markdown files")
    
    # Extract text from all files
    all_text = []
    for file_path in all_files:
        logger.info(f"Processing file: {file_path}")
        if file_path.lower().endswith('.pdf'):
            text = extract_from_pdf(file_path)
        else:  # Markdown files
            text = extract_from_markdown(file_path)
        
        if text:
            # Add file separator for clarity
            all_text.append(f"\n\n--- File: {os.path.basename(file_path)} ---\n\n")
            all_text.append(text)
    
    # Write all text to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_text))
        logger.info(f"Successfully wrote extracted text to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error writing to output file {output_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract text from PDF and Markdown files for model finetuning')
    parser.add_argument('--input_dir', type=str, default='q3_dataset', 
                        help='Directory containing PDF and Markdown files')
    parser.add_argument('--output_file', type=str, default='finetuning_data.txt',
                        help='Output file for extracted text')
    
    args = parser.parse_args()
    
    logger.info(f"Starting extraction from {args.input_dir} to {args.output_file}")
    success = process_files(args.input_dir, args.output_file)
    
    if success:
        logger.info("Extraction completed successfully")
    else:
        logger.error("Extraction failed")

if __name__ == "__main__":
    main()
