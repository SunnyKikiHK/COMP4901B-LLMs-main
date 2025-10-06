import argparse
import regex as re
import requests
import json
from utils import  read_warc_file, read_wet_file
from datasets import load_dataset
from typing import Set, Dict
import string

import chardet
from bs4 import BeautifulSoup
import regex as re
from langdetect import detect_langs, LangDetectException

def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open('./bad_word_list.txt', 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)

def _detect_encoding(html: bytes) -> str:
    """Detects the character encoding of HTML content.
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Detected character encoding (e.g., 'utf-8', 'iso-8859-1').
    """
    if isinstance(html, bytes):
        result = chardet.detect(html)
        return result.get('encoding', 'utf-8') # returns a dictionary containing the estimated encoding, a confidence level, and the language detected.
    return 'utf-8'


def html_to_text(html) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    enc_way = _detect_encoding(html)

    try:
        html = html.decode(enc_way)
    except (UnicodeDecodeError, AttributeError, TypeError):
        html = html.decode('utf-8', errors='ignore') # errors='ignore': If a character or sequence of bytes is encountered that does not conform to the specified encoding, it will be silently dropped from the resulting string or byte sequence.

    result = BeautifulSoup(html, 'html') # lxml是速度较快的解析器，但需要额外安装。html.parser则是Python自带的解析器
    # Remove head, header, and footer sections
    for tag in result(['head', 'header', 'footer', 'script', 'style', 'meta']):
        tag.decompose()  # Remove the tag and its contents
    return result.get_text(separator='\n', strip=True) # get_text()方法用于提取HTML或XML文档中的纯文本内容。separator参数指定文本之间的分隔符
    

def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    # Replace US social security numbers (XXX-XX-XXXX format)
    if not text:
        return text
    
    patterns = {
        'SSN' : r"\b\d{3}-\d{2}-\d{4}\b",
        "US_PHONE": r"\+1\d{10}\b",
    }

    text = re.sub(patterns['SSN'], lambda m: re.sub(r'\d', 'X', m.group()), text)
    # Mask US phone: +1XXXXXXXXXX (preserve +1, replace 10 digits with X)
    text = re.sub(patterns['US_PHONE'], lambda M: '+1' + 'X'*10, text)
    return text
    

def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """
    text = text.replace('\r', '\n') # normalize new line
    
    paragraphs = text.split('\n')
    clean_paragraphs = []

    for para in paragraphs:
        # Drop if >100 alnum chars with no whitespace
        if re.search(r'[A-Za-z0-9]{100,}', para):
            continue
        # Drop if no punctuation
        if not any(ch in string.punctuation for ch in para):
            continue
        clean_paragraphs.append(para)

    text = '\n'.join(clean_paragraphs)
    return text

def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    if not text:
        return False
    
    # presence of bad words 
    tokens = re.split(r'\s+', text.lower())
    bad_word_set = retrieve_bad_words()

    if any(token in bad_word_set for token in tokens):
        return False
    if not any(ch in string.punctuation for ch in text): # At least one punctuation character
        return False
    if not any(not ch.isspace() for ch in text): # Dont want empty or all whitespace
        return False

    valid_chars = sum(ch.isalnum() or ch.isspace() or ch in string.punctuation for ch in text)
    if valid_chars / len(text) < 0.8:
        return False
    return True



def is_english_text(text: str) -> bool:
    """Detects if text is primarily in English based on character distribution.
    Args:
        text (str): Text to analyze
    Returns:
        bool: True if text is primarily English, False otherwise
    """

    # We use two conditions

    try:
        lang_info = detect_langs(text)
        first_cond = lang_info and lang_info[0].lang == 'en' and lang_info[0].prob >= 0.95
    except LangDetectException:
        first_cond = False


    if not text:
        return False

    # Count English letters
    english_letters = re.findall(r'[a-zA-Z]', text)
    
    # Count all alphabetic letters (including non-English)
    total_letters = [char for char in text if char.isalpha()]

    if len(total_letters) == 0:
        return False

    return len(english_letters) / len(total_letters) >= 0.9 and first_cond

    

def deduplicate_texts(texts: list[str]) -> list[str]:
    """Deduplicates text by removing duplicate sentences.
    Args:
        texts (list[str]): List of text strings to deduplicate.
    Returns:
        list[str]: Deduplicated list of texts. Implemented a simple Jaccard similarity based deduplication.
    """
    seen = set()
    deduplicated_texts = [] # unique texts without duplicates
    for text in texts:
        norm_text = text.strip()
        if norm_text not in seen:
            deduplicated_texts.append(norm_text)
            seen.add(norm_text)

    # Jaccard similarity: J(A, B) = A and B / A or B
    jaccard_threshold = 0.5
    def jaccard(a: str, b:str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        if union == 0:
            return 0.0
        return intersection / union
    
    filtered_results = []
    for text in deduplicated_texts:
        can_filter = False
        for filter_text in filtered_results:
            if jaccard(text, filter_text) >= jaccard_threshold:
                can_filter = True
                break
        if not can_filter:
            filtered_results.append(text)
    
    return filtered_results



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str,  default = '', help = 'Specify the path for your warc file.')
    parser.add_argument('--dfname', type = str,  default = '', help = 'Specify the path where you stored topic_dataset.json')
    parser.add_argument('--num_records', type = int,  default=30, help = 'Specify the number of records you want to parse (only used for debugging with smaller sets)')
    parser.add_argument('--output', type = str,  default='cleaned_documents.txt', help = 'Output file for cleaned text documents')
    # parser.add_argument('--wet_name', type = str, default = '', help = 'Specify the path for your wet file.')
    args = parser.parse_args()

    if args.fname:
        seen = 0
        passes = 0

        with open(args.output, 'w', encoding='utf-8') as output_file:
            for url, html_text in read_warc_file(args.fname, args.num_records):
                seen += 1
                # print("Before HTML to text: ", str(html_text))
                text = html_to_text(html_text)
                # print("\n\n\nAfter HTML to text: ", text)
                cleaned_text = clean_text(text)
                # print("After cleaning: ", cleaned_text)
                cleaned_nopii_text = replace_pii(cleaned_text)
                #print("After PII removal: ", cleaned_nopii_text)
                passes_check = heuristic_quality_filter(cleaned_nopii_text)
                is_english = is_english_text(cleaned_nopii_text)
                print(url)
                # print("Passes heuristic quality filter:", passes_check)
                print("Is English text:", is_english)
                if passes_check and is_english:
                    passes += 1
                    # Replace newlines with spaces to keep each document on one line
                    single_line_text = cleaned_nopii_text.replace('\n', ' ').replace('\r', ' ').strip()
                    output_file.write(single_line_text + '\n')
                    print("Saved cleaned English document to output file")
                elif passes_check and not is_english:
                    print("Document filtered out: not English")

        print(f"{passes} passed out of {seen} records processed.")
        print(f"Cleaned documents saved to: {args.output}")

    if args.dfname:
        with open(args.dfname, 'r') as f:
            raw_texts = json.load(f)
        raw_texts = [item['text'] for item in raw_texts['data']]
        deduplicated_texts = deduplicate_texts(raw_texts)
        print(f"{len(deduplicated_texts)} deduplicated out of {len(raw_texts)} records processed.")
    else:
        print("Usage: python homework.py --fname data.warc")