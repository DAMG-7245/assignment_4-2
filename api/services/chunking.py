import os
import json

def chunk_by_fixed_size(text, chunk_size=1000, overlap=200):
    """Chunk text by fixed size with overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Find the nearest period or newline to make chunks end naturally
        if end < len(text):
            for marker in ['. ', '.\n', '\n\n', '\n', ' ']:
                natural_end = text.rfind(marker, start, end)
                if natural_end != -1:
                    end = natural_end + len(marker)
                    break
        
        chunks.append(text[start:end].strip())
        start = max(start, end - overlap)
        
        # Avoid infinite loop for small texts
        if start >= len(text) or end == len(text):
            break
            
    return chunks

def chunk_by_paragraph(text):
    """Chunk text by paragraphs"""
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Further split very long paragraphs
    chunks = []
    for para in paragraphs:
        if len(para) > 1500:
            chunks.extend(chunk_by_fixed_size(para, 1000, 150))
        else:
            chunks.append(para)
            
    return chunks

def chunk_by_semantic_units(text):
    """Chunk text by semantic units (headings, sections, etc.)"""
    # This is a simplified implementation - in real-world you'd use more complex logic
    # Look for section headers, bullet points, etc.
    sections = []
    current_section = []
    
    # Split by lines
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check if line looks like a heading (all caps, ends with :, etc.)
        is_heading = (line.isupper() or line.endswith(':') or 
                     (len(line) < 100 and i < len(lines)-1 and not lines[i+1].strip()))
        
        if is_heading and current_section:
            # End current section and start a new one
            sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    # Add the final section
    if current_section:
        sections.append('\n'.join(current_section))
        
    # Further chunk very long sections
    chunks = []
    for section in sections:
        if len(section) > 1500:
            chunks.extend(chunk_by_fixed_size(section, 1000, 150))
        else:
            chunks.append(section)
            
    return chunks

def chunk_document(parsed_document, chunking_strategy="semantic"):
    """
    Chunk a parsed document using the specified strategy
    
    Args:
        parsed_document: JSON document containing parsed PDF content
        chunking_strategy: Strategy to use for chunking ("fixed", "paragraph", "semantic")
        
    Returns:
        List of chunks with metadata
    """
    # Extract document metadata
    doc_id = parsed_document.get('doc_id', 'unknown')
    year_quarter = parsed_document.get('year_quarter', 'unknown')
    parser = parsed_document.get('parser', 'unknown')
    
    # Concatenate all text from the document
    full_text = ' '.join([page['text'] for page in parsed_document['content']])
    
    # Apply the selected chunking strategy
    if chunking_strategy == "fixed":
        chunks = chunk_by_fixed_size(full_text)
    elif chunking_strategy == "paragraph":
        chunks = chunk_by_paragraph(full_text)
    elif chunking_strategy == "semantic":
        chunks = chunk_by_semantic_units(full_text)
    else:
        # Default to semantic chunking
        chunks = chunk_by_semantic_units(full_text)
    
    # Create chunk objects with metadata
    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            'id': f"{doc_id}_{chunking_strategy}_{i}",
            'text': chunk_text,
            'metadata': {
                'doc_id': doc_id,
                'year_quarter': year_quarter,
                'chunk_strategy': chunking_strategy,
                'chunk_index': i,
                'parser': parser
            }
        })
    
    return result