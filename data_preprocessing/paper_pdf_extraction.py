import os
import glob
import re
import json
import argparse
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    import PyPDF2
    logger.info("Using PyPDF2 for PDF extraction")
except ImportError:
    logger.warning("PyPDF2 not found, attempting to install...")
    import subprocess
    subprocess.check_call(["pip", "install", "PyPDF2"])
    import PyPDF2
    logger.info("PyPDF2 installed successfully")

try:
    from transformers import AutoTokenizer
    logger.info("Using HuggingFace Transformers for tokenization")
except ImportError:
    logger.warning("transformers not found, attempting to install...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers"])
    from transformers import AutoTokenizer
    logger.info("transformers installed successfully")

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num} in {pdf_path}: {e}")
                    continue
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def clean_text(text):
    if not text:
        return ""
    
    try:
        text = text.encode('utf-8', 'replace').decode('utf-8')
        
        text = re.sub(r'\n+', '\n', text)
        
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        lines = text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            if len(line.strip()) < 3:
                continue
            if re.match(r'^\s*\d+\s*$', line):
                continue
            try:
                cleaned_line = ''.join(c if ord(c) < 65536 else '?' for c in line)
                cleaned_lines.append(cleaned_line)
            except Exception as e:
                logger.warning(f"Error cleaning line: {e}")
                continue
        
        text = '\n'.join(cleaned_lines)
        
        text = re.sub(r'\s+', ' ', text)
        
        try:
            text = re.split(r'References|REFERENCES|Bibliography|BIBLIOGRAPHY', text)[0]
        except Exception as e:
            logger.warning(f"Error removing references section: {e}")
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        try:
            return text.encode('ascii', 'ignore').decode('ascii')
        except:
            return ""

def process_pdf(pdf_path):
    try:
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing {filename}")
        
        raw_text = extract_text_from_pdf(pdf_path)
        
        cleaned_text = clean_text(raw_text)
        
        return {
            "filename": filename,
            "path": pdf_path,
            "text": cleaned_text,
            "category": os.path.basename(os.path.dirname(pdf_path))
        }
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None

def process_pdfs_in_directory(directory, output_file, max_workers=None):
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    pdf_files = glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True)
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_pdf, pdf_files), total=len(pdf_files)):
            if result and result.get("text"):
                results.append(result)
    
    logger.info(f"Successfully processed {len(results)} PDF files")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            try:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.warning(f"Error writing result for {item.get('filename')}: {e}")
                try:
                    item['text'] = item['text'].encode('ascii', 'ignore').decode('ascii')
                    f.write(json.dumps(item) + '\n')
                except Exception as e2:
                    logger.error(f"Failed to write result even with ASCII encoding: {e2}")
    
    logger.info(f"Results saved to {output_file}")
    return results

def create_training_file(jsonl_file, output_file, tokenizer_name="Qwen/Qwen2.5-3B-Instruct"):
    logger.info(f"Creating training file from {jsonl_file}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logger.warning(f"Error loading tokenizer: {e}. Proceeding without tokenization.")
        tokenizer = None
    
    texts = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("text"):
                    text = item["text"]
                    text = ''.join(c if ord(c) < 65536 else '' for c in text)
                    texts.append(text)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing JSON line: {e}")
            except Exception as e:
                logger.warning(f"Error processing line: {e}")
    
    logger.info(f"Loaded {len(texts)} text samples")
    
    combined_text = "\n\n".join(texts)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        logger.info(f"Training file saved to {output_file}")
    except UnicodeEncodeError as e:
        logger.warning(f"Unicode encoding error: {e}. Trying with error handling...")
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            f.write(combined_text)
        logger.info(f"Training file saved to {output_file} with character replacement")
    except Exception as e:
        logger.error(f"Error saving training file: {e}")
        try:
            ascii_text = combined_text.encode('ascii', 'ignore').decode('ascii')
            with open(output_file, 'w', encoding='ascii') as f:
                f.write(ascii_text)
            logger.info(f"Training file saved to {output_file} with ASCII encoding (some characters lost)")
        except Exception as e2:
            logger.error(f"Failed to save training file even with ASCII encoding: {e2}")
            return False
    
    if tokenizer:
        try:
            tokens = tokenizer.encode(combined_text[:1000000])
            estimated_tokens = len(tokens) * (len(combined_text) / 1000000) if len(combined_text) > 1000000 else len(tokens)
            logger.info(f"Training file created with approximately {int(estimated_tokens)} tokens")
        except Exception as e:
            logger.warning(f"Error calculating token count: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract and preprocess text from PDF files for finetuning")
    parser.add_argument("--input_dir", type=str, default="papers", help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory to save processed data")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    jsonl_file = os.path.join(args.output_dir, "processed_pdfs.jsonl")
    process_pdfs_in_directory(args.input_dir, jsonl_file, max_workers=args.workers)
    
    training_file = os.path.join(args.output_dir, "training_data.txt")
    success = create_training_file(jsonl_file, training_file)
    
    if not success:
        logger.warning("Failed to create training file with all data. Creating a simplified version...")
        simplified_file = os.path.join(args.output_dir, "training_data_simplified.txt")
        with open(jsonl_file, 'r', encoding='utf-8') as f, open(simplified_file, 'w', encoding='utf-8') as out_f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("text"):
                        text = item["text"].encode('ascii', 'ignore').decode('ascii')
                        out_f.write(text + "\n\n")
                except Exception:
                    continue
        logger.info(f"Simplified training file saved to {simplified_file}")

if __name__ == "__main__":
    main() 