import os
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Try to use lxml for better XML support, fall back to ElementTree
try:
    from lxml import etree as ET
    LXML_AVAILABLE = True
except ImportError:
    import xml.etree.ElementTree as ET
    LXML_AVAILABLE = False
    print("Warning: lxml not available. Using xml.etree.ElementTree (some features may be limited)")

class SanskritTextProcessor:
    """Process GRETIL XML files to extract quotes and metadata"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Extract metadata from filename"""
        # Remove .xml extension
        base_name = filename.replace('.xml', '')
        
        # Split by underscores and hyphens to extract components
        parts = base_name.replace('-', '_').split('_')
        
        # First part is usually language (sa = Sanskrit)
        language = parts[0] if parts else 'unknown'
        
        # Try to extract author and work
        author = 'unknown'
        work = 'unknown'
        
        # Clean up work name to single word
        if len(parts) > 1:
            # Common pattern: sa_author-work or sa_author_work
            remaining = '_'.join(parts[1:])
            
            # Try to split on common patterns
            if '-' in remaining:
                author_work = remaining.split('-', 1)
                author = author_work[0]
                work = author_work[1].split()[0] if len(author_work) > 1 else 'unknown'  # Take first word only
            else:
                # If no clear separator, take first part as author
                author = parts[1]
                work_parts = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                work = work_parts.split()[0] if work_parts != 'unknown' else 'unknown'  # Take first word only
        
        return {
            'language': language,
            'author': author,
            'work': work,
            'filename': filename
        }
    
    def extract_text_segments(self, xml_path: Path) -> List[Dict]:
        """Extract text segments from XML file with hierarchical structure"""
        try:
            # Try different parsing approaches
            tree = None
            
            if LXML_AVAILABLE:
                try:
                    # Try lxml first (more robust)
                    parser = ET.XMLParser(recover=True, encoding='utf-8')
                    tree = ET.parse(str(xml_path), parser)
                except Exception as e:
                    print(f"lxml parsing failed for {xml_path.name}: {e}")
            
            if tree is None:
                # Fallback to ElementTree
                try:
                    tree = ET.parse(xml_path)
                except ET.ParseError as e:
                    print(f"XML parsing error in {xml_path.name}: {e}")
                    return []
                except Exception as e:
                    print(f"Unexpected error parsing {xml_path.name}: {e}")
                    return []
            
            root = tree.getroot()
            
            segments = []
            file_metadata = self.parse_filename(xml_path.name)
            
            # Extract title from header if available
            title_elem = root.find('.//tei:title', self.namespaces)
            if title_elem is not None and title_elem.text:
                # Take only first word of work title
                full_title = title_elem.text.strip()
                file_metadata['work'] = full_title.split()[0] if full_title else 'unknown'
            
            # Extract author from header if available
            author_elem = root.find('.//tei:author', self.namespaces)
            if author_elem is not None and author_elem.text:
                file_metadata['author'] = author_elem.text.strip()
            
            # Find all verse/line elements
            verses = root.findall('.//tei:lg', self.namespaces)  # verse groups
            lines = root.findall('.//tei:l', self.namespaces)    # individual lines
            paragraphs = root.findall('.//tei:p', self.namespaces)  # paragraphs
            
            # Process verse groups
            for i, lg in enumerate(verses):
                verse_id = lg.get('{http://www.w3.org/XML/1998/namespace}id', f'verse_{i+1}')
                
                # Extract all lines in this verse group
                verse_lines = lg.findall('.//tei:l', self.namespaces)
                verse_text = []
                
                for line in verse_lines:
                    if line.text and line.text.strip():
                        verse_text.append(line.text.strip())
                
                if verse_text:
                    full_verse = ' / '.join(verse_text)  # Sanskrit verses often separated by /
                    
                    segments.append({
                        'text': full_verse,
                        'type': 'verse',
                        'id': verse_id,
                        'chapter': self._extract_chapter_info(lg, root),
                        'metadata': file_metadata.copy()
                    })
            
            # Process standalone lines (not in verse groups)
            processed_line_ids = set()
            for lg in verses:
                for line in lg.findall('.//tei:l', self.namespaces):
                    line_id = line.get('{http://www.w3.org/XML/1998/namespace}id')
                    if line_id:
                        processed_line_ids.add(line_id)
            
            for i, line in enumerate(lines):
                line_id = line.get('{http://www.w3.org/XML/1998/namespace}id')
                if line_id not in processed_line_ids and line.text and line.text.strip():
                    segments.append({
                        'text': line.text.strip(),
                        'type': 'line',
                        'id': line_id or f'line_{i+1}',
                        'chapter': self._extract_chapter_info(line, root),
                        'metadata': file_metadata.copy()
                    })
            
            # Process paragraphs
            for i, p in enumerate(paragraphs):
                if p.text and p.text.strip() and len(p.text.strip()) > 20:  # Only meaningful paragraphs
                    segments.append({
                        'text': p.text.strip(),
                        'type': 'paragraph',
                        'id': f'para_{i+1}',
                        'chapter': self._extract_chapter_info(p, root),
                        'metadata': file_metadata.copy()
                    })
            
            return segments
            
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            return []
    
    def _extract_chapter_info(self, element, root=None) -> Dict[str, str]:
        """Extract chapter/section information from element context"""
        chapter_info = {
            'book': 'unknown',
            'chapter': 'unknown',
            'section': 'unknown'
        }
        
        if LXML_AVAILABLE:
            # Use lxml's getparent() method
            parent = element.getparent()
            while parent is not None:
                if parent.tag.endswith('div'):
                    div_type = parent.get('type', '')
                    div_n = parent.get('n', '')
                    
                    if 'book' in div_type.lower() or 'adhyaya' in div_type.lower():
                        chapter_info['book'] = div_n or div_type
                    elif 'chapter' in div_type.lower() or 'paricchedika' in div_type.lower():
                        chapter_info['chapter'] = div_n or div_type
                    elif 'section' in div_type.lower():
                        chapter_info['section'] = div_n or div_type
                
                parent = parent.getparent()
        else:
            # Fallback method for ElementTree: search for div ancestors
            if root is not None:
                # Find all div elements and check if they contain our element
                divs = root.findall('.//tei:div', self.namespaces)
                for div in divs:
                    if self._is_ancestor(div, element):
                        div_type = div.get('type', '')
                        div_n = div.get('n', '')
                        
                        if 'book' in div_type.lower() or 'adhyaya' in div_type.lower():
                            chapter_info['book'] = div_n or div_type
                        elif 'chapter' in div_type.lower() or 'paricchedika' in div_type.lower():
                            chapter_info['chapter'] = div_n or div_type
                        elif 'section' in div_type.lower():
                            chapter_info['section'] = div_n or div_type
        
        return chapter_info
    
    def _is_ancestor(self, ancestor, descendant) -> bool:
        """Check if ancestor contains descendant (ElementTree fallback)"""
        for elem in ancestor.iter():
            if elem == descendant:
                return True
        return False

def generate_quote_identification_dataset(data_path: str, 
                                        min_quote_length: int = 10,
                                        max_quote_length: int = 200,
                                        num_samples: int = 1000) -> List[Dict]:
    """Generate dataset for Sanskrit quote identification task"""
    
    processor = SanskritTextProcessor(data_path)
    data_dir = Path(data_path)
    
    # Find all XML files and limit to 10
    xml_files = list(data_dir.glob('*.xml'))[:10]  # Hard limit to 10 files
    print(f"Found {len(xml_files)} XML files, processing first 10")
    
    all_segments = []
    
    # Process each XML file
    for xml_file in xml_files:
        print(f"Processing {xml_file.name}...")
        segments = processor.extract_text_segments(xml_file)
        all_segments.extend(segments)
    
    print(f"Extracted {len(all_segments)} text segments total")
    
    # Filter segments by length
    valid_segments = [
        seg for seg in all_segments 
        if min_quote_length <= len(seg['text']) <= max_quote_length
    ]
    
    print(f"Found {len(valid_segments)} segments within length range")
    
    # Sample quotes for dataset
    if len(valid_segments) > num_samples:
        sampled_segments = random.sample(valid_segments, num_samples)
    else:
        sampled_segments = valid_segments
    
    # System message for the task
    system_message = """You are an expert Sanskrit librarian and scholar. Your task is to identify the source of Sanskrit text quotes from the GRETIL digital library corpus.

Given a Sanskrit quote, you must identify:
1. The author (if known)
2. The work/text title
3. The book/adhyāya (if applicable)
4. The chapter/section (if applicable)
5. The verse/line number (if applicable)

Provide your answer in JSON format:
{
    "author": "author_name",
    "work": "work_title", 
    "book": "book_number_or_name",
    "chapter": "chapter_number_or_name",
    "verse": "verse_or_line_identifier",
    "confidence": 0.95
}

Use "unknown" for any field you cannot determine. Set confidence between 0.0 and 1.0 based on how certain you are of your identification.

Now identify the source of this Sanskrit quote:"""

    jsonl_entries = []
    
    for segment in sampled_segments:
        # Create user input with the quote
        user_input = f'Sanskrit quote: "{segment["text"]}"'
        
        # Create expected answer with cleaned verse number
        expected_answer = {
            "author": segment['metadata']['author'],
            "work": segment['metadata']['work'],
            "book": segment['chapter']['book'],
            "chapter": segment['chapter']['chapter'],
            "verse": extract_verse_number(segment['id']),
            "confidence": 1.0
        }
        
        # Determine difficulty level
        difficulty = determine_difficulty(
            len(segment['text']),
            segment['metadata']['author'],
            segment['metadata']['work']
        )
        
        # Create JSONL entry
        jsonl_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": user_input
                }
            ],
            "quote": segment['text'],
            "quote_type": segment['type'],
            "difficulty": difficulty,
            "expected_answer": expected_answer,
            "metadata": {
                "filename": segment['metadata']['filename'],
                "segment_id": segment['id'],
                "chapter_info": segment['chapter'],
                "text_length": len(segment['text'])
            }
        }
        
        jsonl_entries.append(jsonl_entry)
    
    return jsonl_entries

def extract_verse_number(verse_id: str) -> str:
    """Extract only the numeric part from verse ID"""
    if not verse_id:
        return "0"
    
    # Use regex to find all numbers in the string
    numbers = re.findall(r'\d+', verse_id)
    
    # If we found numbers, return the first one, otherwise return "0"
    return numbers[0] if numbers else "0"

def determine_difficulty(quote_length: int, author: str, work: str) -> str:
    """Determine difficulty level based on quote characteristics"""
    # Easy: well-known authors, longer quotes
    if (author.lower() in ['abhinavagupta', 'kalidasa', 'bhartrhari', 'nagarjuna'] or
        quote_length > 100):
        return 'easy'
    # Hard: unknown/generic metadata, very short quotes  
    elif (author == 'unknown' or work == 'unknown' or quote_length < 30):
        return 'hard'
    # Medium: everything else
    else:
        return 'medium'

def write_jsonl_file(data: List[Dict], filename: str):
    """Write data to JSONL file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def split_dataset(data: List[Dict], train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test"""
    random.seed(42)
    
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "./gretil_data/"  # Path to your XML files
    NUM_SAMPLES = 2000
    MIN_QUOTE_LENGTH = 15
    MAX_QUOTE_LENGTH = 300
    
    # Check if data path exists
    if not os.path.exists(DATA_PATH):
        print(f"Data path {DATA_PATH} does not exist. Please update the path to your GRETIL XML files.")
        exit(1)
    
    # Create output directory
    output_dir = Path("sanskrit_dataset_output")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating Sanskrit quote identification dataset...")
    
    # Generate dataset
    dataset = generate_quote_identification_dataset(
        DATA_PATH, 
        min_quote_length=MIN_QUOTE_LENGTH,
        max_quote_length=MAX_QUOTE_LENGTH,
        num_samples=NUM_SAMPLES
    )
    
    print(f"Generated {len(dataset)} training examples")
    
    # Count difficulty distribution
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    for entry in dataset:
        difficulty_counts[entry['difficulty']] += 1
    
    print(f"Difficulty distribution - Easy: {difficulty_counts['easy']}, "
          f"Medium: {difficulty_counts['medium']}, Hard: {difficulty_counts['hard']}")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Create human-readable timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Write files to output directory
    write_jsonl_file(train_data, output_dir / f"sanskrit_quote_id_train_{timestamp}.jsonl")
    write_jsonl_file(val_data, output_dir / f"sanskrit_quote_id_val_{timestamp}.jsonl") 
    write_jsonl_file(test_data, output_dir / f"sanskrit_quote_id_test_{timestamp}.jsonl")
    
    # Write complete dataset
    write_jsonl_file(dataset, output_dir / f"sanskrit_quote_id_complete_{timestamp}.jsonl")
    
    print(f"\nDataset files created in '{output_dir}':")
    print(f"  Training: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples") 
    print(f"  Test: {len(test_data)} examples")
    print(f"  Complete: {len(dataset)} examples")
    print(f"  Timestamp: {timestamp}")
    
    # Print sample entry
    if dataset:
        print(f"\nSample entry:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
