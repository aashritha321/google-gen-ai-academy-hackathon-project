import os
import time
import random
import json
import numpy as np
import faiss
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

# Load env variables from .env
load_dotenv()

def extract_text_and_transitions(pdf_path):
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    full_text = ""
    page_transitions = []
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            page_text = ""
        start_idx = len(full_text)
        full_text += page_text + "\n"
        end_idx = len(full_text)
        page_transitions.append({
            "start": start_idx,
            "end": end_idx,
            "page_num": i + 1
        })
    print(f"Extracted {len(full_text)} characters and mapped {len(page_transitions)} pages.")
    return full_text, page_transitions

def recursive_chunk_text(text, chunk_size=6000, chunk_overlap=1000):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        if end < text_len:
            lookback_start = max(start, end - 800)
            search_area = text[lookback_start:end]
            
            best_break = -1
            for separator in [".\n", "\n", ". ", " "]:
                idx = search_area.rfind(separator)
                if idx != -1:
                    best_break = lookback_start + idx + len(separator)
                    break
            
            if best_break != -1:
                end = best_break
        
        chunk_str = text[start:end]
        if chunk_str.strip():
            chunks.append((chunk_str, start, end))
            
        if end >= text_len:
            break
            
        start = max(start + 1, end - chunk_overlap)
        
    return chunks

def split_text_into_chunks(full_text, page_transitions, chunk_size=6000, chunk_overlap=1000):
    chunks = []
    paragraphs = full_text.split("\n\n")
    current_char_idx = 0
    
    for para in paragraphs:
        if not para.strip():
            continue
        para_start = full_text.find(para, current_char_idx)
        if para_start == -1:
            para_start = current_char_idx
        
        para_end = para_start + len(para)
        current_char_idx = para_end + 2
        
        if len(para) <= chunk_size:
            chunks.append((para, para_start, para_end))
        else:
            sub_chunks = recursive_chunk_text(para, chunk_size, chunk_overlap)
            for sub_text, sub_start_offset, sub_end_offset in sub_chunks:
                chunks.append((sub_text, para_start + sub_start_offset, para_start + sub_end_offset))
                
    return chunks

def get_pages_for_chunk(chunk_start, chunk_end, page_transitions):
    pages = []
    for trans in page_transitions:
        overlap_start = max(chunk_start, trans['start'])
        overlap_end = min(chunk_end, trans['end'])
        if overlap_start < overlap_end:
            pages.append(trans['page_num'])
    return pages if pages else [1]

def embed_text_with_retry(client, model, texts, max_retries=10, initial_backoff=5.0):
    from google.genai import types
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
            response = client.models.embed_content(
                model=model,
                contents=contents
            )
            return response
        except Exception as e:
            err_msg = str(e)
            if attempt == max_retries - 1:
                raise e
            sleep_time = backoff + random.uniform(0, 1.0)
            print(f"Embedding API error (attempt {attempt+1}/{max_retries}): {err_msg}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            backoff *= 2

def main():
    # Find the legal_docs directory relative to working folder
    possible_paths = [
        "legal_docs",
        "../legal_docs",
        "nyayamitra/legal_docs"
    ]
    docs_dir = None
    for p in possible_paths:
        if os.path.isdir(p):
            docs_dir = p
            break
            
    if not docs_dir:
        print("Error: Could not locate 'legal_docs' directory containing PDFs.")
        print("Please ensure it exists in the workspace.")
        return
        
    print(f"Found legal docs directory at: {docs_dir}")
    
    # Find all PDF files
    pdf_files = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the legal docs directory.")
        return
        
    print(f"Processing {len(pdf_files)} PDF documents: {pdf_files}")
    
    # split text into chunks
    all_chunks_metadata = []
    for pdf_path in pdf_files:
        full_text, page_transitions = extract_text_and_transitions(pdf_path)
        chunks = split_text_into_chunks(full_text, page_transitions, chunk_size=8000, chunk_overlap=1500)
        
        doc_name = os.path.basename(pdf_path)
        for idx, (chunk_text, start_offset, end_offset) in enumerate(chunks):
            pages = get_pages_for_chunk(start_offset, end_offset, page_transitions)
            all_chunks_metadata.append({
                "id": f"{doc_name}_chunk_{idx}",
                "doc_name": doc_name,
                "source": doc_name,
                "text": chunk_text,
                "pages": pages,
                "start_offset": start_offset,
                "end_offset": end_offset
            })
            
    print(f"Total chunks across all documents: {len(all_chunks_metadata)}")
    
    # Initialize Google GenAI client
    client = genai.Client()
    
    # We use gemini-embedding-001 as the modern embedding model
    model = "gemini-embedding-001"
    
    # Generate embeddings in batches with retry logic
    batch_size = 5
    embeddings = []
    
    print("Generating embeddings using gemini-embedding-001...")
    for i in range(0, len(all_chunks_metadata), batch_size):
        batch = all_chunks_metadata[i:i+batch_size]
        batch_texts = [item["text"] for item in batch]
        
        print(f"Embedding batch {i//batch_size + 1}/{(len(all_chunks_metadata)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        response = embed_text_with_retry(client, model, batch_texts)
        for emb in response.embeddings:
            embeddings.append(emb.values)
            
        # Polite pause to stay safely below 100 Requests Per Minute (RPM) free-tier limit (5 chunks per batch * 12 batches/min = 60 RPM)
        time.sleep(5.0)
        
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings_arr.shape}")
    
    # Normalize vectors for Cosine Similarity (IndexFlatIP)
    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
    normalized_embeddings = embeddings_arr / (norms + 1e-9)
    
    # Create FAISS index
    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)
    
    # Save FAISS index and metadata
    vector_store_dir = "vector_store"
    # Fallback to local subdirectory if running from nyayamitra
    if not os.path.exists(vector_store_dir) and os.path.basename(os.getcwd()) != "nyayamitra":
        vector_store_dir = "nyayamitra/vector_store"
        
    os.makedirs(vector_store_dir, exist_ok=True)
    
    index_path = os.path.join(vector_store_dir, "index.faiss")
    metadata_path = os.path.join(vector_store_dir, "metadata.json")
    
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks_metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved FAISS index to {index_path}")
    print(f"Successfully saved metadata to {metadata_path}")
    print("Indexing completed successfully!")

if __name__ == "__main__":
    main()
