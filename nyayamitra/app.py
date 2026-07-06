import os
import re
import json
import numpy as np
import faiss
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Set page configuration for layout and title
st.set_page_config(
    page_title="NyayaMitra AI — Legal Decision Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Inject modern, premium custom CSS (Dark Mode and Glassmorphism aesthetics)

        font-size: 0.st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFB703, #FB8500, #219EBC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #8A9CA8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card design */
    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .premium-card:hover {
        border-color: rgba(255, 183, 3, 0.4);
        transform: translateY(-2px);
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFB703;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chips and tags */
    .legal-chip {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .chip-criminal { background-color: #d90429; }
    .chip-civil { background-color: #2a9d8f; }
    .chip-cyber { background-color: #4361ee; }
    .chip-family { background-color: #7209b7; }
    .chip-constitutional { background-color: #f77f00; }
    .chip-default { background-color: #6c757d; }
    
    /* Timeline */
    .timeline {
        border-left: 2px solid rgba(255, 255, 255, 0.1);
        padding-left: 1.2rem;
        margin-left: 0.5rem;
    }
    .timeline-item {
        position: relative;
        margin-bottom: 1rem;
    }
    .timeline-marker {
        position: absolute;
        left: -1.65rem;
        top: 0.2rem;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #FFB703;
        border: 2px solid #0E1117;
    }
    .timeline-text {95rem;
        line-height: 1.4;
    }
    
    /* Resource cards */
    .resource-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 1rem;
    }
    .resource-card {
        background: rgba(33, 158, 188, 0.15);
        border: 1px solid rgba(33, 158, 188, 0.3);
        border-radius: 8px;
        padding: 0.85rem;
        text-align: center;
    }
    .resource-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #219EBC;
        margin-bottom: 0.25rem;
    }
    .resource-number {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Disclaimer alert */
    .disclaimer-alert {
        background: rgba(220, 53, 69, 0.1);
        border: 1px solid rgba(220, 53, 69, 0.3);
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #ea868f;
        margin-top: 1.5rem;
        display: flex;
        gap: 0.5rem;
    }
    
    /* Source passages */
    .source-block {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 6px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #8A9CA8;
    }
    .source-meta {
        font-size: 0.8rem;
        color: #8A9CA8;
        margin-bottom: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache index loading to ensure Streamlit doesn't reload FAISS on every interaction
@st.cache_resource
def load_vector_store():
    # Detect the vector store folder paths
    vector_store_dir = "vector_store"
    if not os.path.exists(vector_store_dir) and os.path.exists("nyayamitra/vector_store"):
        vector_store_dir = "nyayamitra/vector_store"
        
    index_path = os.path.join(vector_store_dir, "index.faiss")
    metadata_path = os.path.join(vector_store_dir, "metadata.json")
    
    # Load FAISS index
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, []
    index = faiss.read_index(index_path)
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # Self-healing check: slice metadata to match the number of vectors in FAISS index if building is in progress
    if len(metadata) > index.ntotal:
        metadata = metadata[:index.ntotal]
    return index, metadata

# Initialize the genai client
@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

# Function to get query embedding vector
def get_query_embedding(client, text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

# Perform similarity search against loaded FAISS index
def retrieve_relevant_documents(client, index, metadata, query, k=5):
    query_vector = np.array([get_query_embedding(client, query)], dtype=np.float32)
    
    # Normalize query vector (since build_index.py saved normalized Inner Product vectors)
    norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    normalized_query = query_vector / (norm + 1e-9)
    
    # Perform FAISS search
    distances, indices = index.search(normalized_query, k)
    
    results = []
    for i in range(k):
        idx = indices[0][i]
        score = float(distances[0][i]) # Cosine similarity score in range [-1, 1]
        if idx < len(metadata):
            item = metadata[idx]
            results.append({
                "text": item["text"],
                "source": item["source"],
                "pages": item["pages"],
                "score": score
            })
    return results

# Transcribe/summarize uploaded images or PDFs using Gemini
def process_multimodal_file(client, uploaded_file):
    # Check mime type
    mime_type = uploaded_file.type
    
    with st.spinner("Processing uploaded document with Gemini's native document understanding..."):
        try:
            if mime_type.startswith("image/"):
                img = Image.open(uploaded_file)
                prompt = (
                    "Analyze this uploaded image of a notice, FIR, letter, or legal document. "
                    "Extract and summarize the core legal facts, names, dates, sections, and the primary legal issue "
                    "described. Provide a detailed summary in plain English so it can be searched against a legal index. "
                    "Only output the summary of the legal issue and facts."
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, img]
                )
                return response.text
                
            elif mime_type == "application/pdf":
                pdf_bytes = uploaded_file.read()
                prompt = (
                    "Analyze this uploaded PDF legal document. "
                    "Extract and summarize the core legal facts, names, dates, sections, and the primary legal issue "
                    "described. Provide a detailed summary in plain English so it can be searched against a legal index. "
                    "Only output the summary of the legal issue and facts."
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                    ]
                )
                return response.text
            else:
                st.error("Unsupported file type. Please upload PNG, JPG, or PDF.")
                return None
        except Exception as e:
            st.error(f"Error parsing document: {e}")
            return None

# Generate structured legal answer using the pipeline
def generate_legal_answer(client, user_query, retrieved_docs, enable_web_search=False):
    # Format retrieved passages as context
    context_text = ""
    for i, doc in enumerate(retrieved_docs):
        context_text += f"--- Passage {i+1} ---\n"
        context_text += f"Source: {doc['source']}, Pages: {doc['pages']}, Search Similarity Score: {doc['score']:.4f}\n"
        context_text += f"Content: {doc['text']}\n\n"
        
    # Standard prompt for structured output
    system_instruction = """
You are "NyayaMitra AI", a Legal Decision Intelligence Platform helping common Indian citizens understand their legal rights and next steps.

IMPORTANT: You must write your entire response in the SAME language as the user's query (e.g., if the user asks in Hindi, respond in Hindi; if in Telugu, respond in Telugu; if in English, respond in English).

Your response must follow this EXACT structure and order, using the exact headers. Ground all cited legal authorities STRICTLY in the provided retrieved context and the live web search grounding results (if available). Do not guess or hallucinate citations.

---

### 1. ISSUE CLASSIFICATION
Classify the type of legal issue (e.g., Criminal, Civil, Cyber Crime, Family Law, Constitutional).

### 2. APPLICABLE LEGAL AUTHORITY
List the exact article or section numbers from the retrieved legal context or live web search grounding results ONLY (referencing BNS, BNSS, BSA, or the Constitution). If no relevant legal authorities are found in the retrieved context or live search, state clearly: "No direct applicable authorities found."

### 3. SIMPLE EXPLANATION
Explain what the laws mean and how they apply to the user's issue in plain, simple, everyday language. Avoid complex legalese.


### 4. STEP-BY-STEP ACTION PLAN

Return ONLY plain text.

Example:

1. File an FIR.
2. Block your SIM.
3. Visit the CEIR portal.
4. Preserve purchase proof.

DO NOT use HTML.
DO NOT use Markdown.
DO NOT use XML.
DO NOT use code blocks.
DO NOT use <div>, <strong>, <span> or any tags.


### 5. CONFIDENCE LEVEL
Output a rating: **High Confidence**, **Medium Confidence**, or **Low Confidence**. Ground this rating explicitly on the provided Search Similarity Scores and Web Grounding accuracy. If similarity scores of the retrieved context are low (e.g., all below 0.60) and search results are sparse, rate as Low Confidence. If there is a high-scoring direct match, rate as High Confidence. Briefly explain your rating based on these signals.

### 6. RELEVANT EMERGENCY RESOURCES
If applicable to the issue classification, list key citizen helpline contacts and portals (e.g., National Consumer Helpline: 1915, Cyber Crime: 1930, Women Helpline: 1091, Police Emergency: 112). Only display this section if relevant.

### 7. DISCLAIMER
"Disclaimer: This is an AI-powered assistant designed for informational purposes. It is not a substitute for professional legal counsel. Please consult a qualified lawyer for official legal advice."
"""

    prompt = f"""
User's Legal Query/Issue Description:
{user_query}

Retrieved Legal Context (from FAISS):
{context_text}

Generate the response adhering strictly to the system instruction, in the user's query language.
"""

    # Configure Google Search grounding tool if requested and enabled
    config_tools = [{"google_search": {}}] if enable_web_search else None

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            tools=config_tools,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )
    )
    
    grounding_metadata = None
    if enable_web_search and response.candidates and response.candidates[0].grounding_metadata:
        grounding_metadata = response.candidates[0].grounding_metadata
    
    print("DEBUG: generate_legal_answer - enable_web_search =", enable_web_search)
    print("DEBUG: generate_legal_answer - grounding_metadata =", grounding_metadata)
    if grounding_metadata:
        queries = getattr(grounding_metadata, "web_search_queries", []) or []
        chunks = getattr(grounding_metadata, "grounding_chunks", []) or []
        print("DEBUG: web_search_queries =", queries)
        print("DEBUG: grounding_chunks count =", len(chunks))
        
    return response.text, grounding_metadata

# Helper to format the final markdown response into premium HTML blocks
def parse_markdown_to_premium_ui(answer_text, retrieved_docs, grounding_metadata=None):
    if not answer_text:
        st.error("⚠️ NyayaMitra was unable to generate a response for this query. This can happen if the content triggers AI safety policies or if there is a transient API issue. Please try rephrasing your description (e.g., focus on the legal facts).")
        return
        
     # Split by headers to extract individual sections
    headers = [
        "1. ISSUE CLASSIFICATION",
        "2. APPLICABLE LEGAL AUTHORITY",
        "3. SIMPLE EXPLANATION",
        "4. STEP-BY-STEP ACTION PLAN",
        "5. CONFIDENCE LEVEL",
        "6. RELEVANT EMERGENCY RESOURCES",
        "7. DISCLAIMER"
    ]

    content_blocks = {}

    for i, header in enumerate(headers):
        pattern = re.escape(header)

        if i + 1 < len(headers):
            next_pattern = re.escape(headers[i + 1])
            match = re.search(
                pattern + r"(.*?)" + next_pattern,
                answer_text,
                flags=re.DOTALL,
            )
        else:
            match = re.search(
                pattern + r"(.*)",
                answer_text,
                flags=re.DOTALL,
            )

        if match:
            content_blocks[header] = match.group(1).strip()
        else:
            content_blocks[header] = ""

    # Clean up formatting
    for key in content_blocks:
        content_blocks[key] = (
        content_blocks[key]
        .replace("###", "")
        .replace("##", "")
        .replace("#", "")
        .lstrip(":\n -*")
        .strip()
    )
    # --- Render Issue Classification ---
    issue_class = content_blocks.get("1. ISSUE CLASSIFICATION", "General")
    chip_class = "chip-default"
    lower_class = issue_class.lower()
    if "criminal" in lower_class:
        chip_class = "chip-criminal"
    elif "civil" in lower_class:
        chip_class = "chip-civil"
    elif "cyber" in lower_class:
        chip_class = "chip-cyber"
    elif "family" in lower_class:
        chip_class = "chip-family"
    elif "constitution" in lower_class:
        chip_class = "chip-constitutional"
        
    st.markdown(f"""    
    
    <div class="premium-card">
        <div class="card-header">⚖️ Issue Classification</div>
        <span class="legal-chip {chip_class}">{issue_class}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Render Legal Authority ---
    auth_text = content_blocks.get("2. APPLICABLE LEGAL AUTHORITY", "")
    st.markdown(f"""
    <div class="premium-card">
        <div class="card-header">📜 Applicable Legal Authority</div>
        <div style="font-size: 1rem; line-height: 1.5; color:var(--text-color);">
            {auth_text.replace('\n', '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Render Explanation ---
    expl_text = content_blocks.get("3. SIMPLE EXPLANATION", "")
    st.markdown(f"""
    <div class="premium-card">
        <div class="card-header">📖 Simple Explanation</div>
        <div style="font-size: 1rem; line-height: 1.6; color: var(--text-color);">
            {expl_text.replace('\n', '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Render Action Plan ---
    plan_text = content_blocks.get("4. STEP-BY-STEP ACTION PLAN", "")
    plan_text = re.sub(r"<[^>]+>", "", plan_text)

    # Convert markdown bullets/numbers into timeline items
    timeline_html = ""

    for line in plan_text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Skip markdown headings
        if line.startswith("#"):
            continue

        # Remove markdown bullets/numbers
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)

        # Convert bold markdown to HTML
        line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)

        timeline_html += (
            f'<div class="timeline-item">'
            f'<div class="timeline-marker"></div>'
            f'<div class="timeline-text">{line}</div>'
            f'</div>'
        )

    action_plan_inner = timeline_html if timeline_html else plan_text.replace("\n", "<br>")
    st.markdown(
        f'<div class="premium-card"><div class="card-header">🎯 Step-by-Step Action Plan</div>'
        f'<div class="timeline">{action_plan_inner}</div></div>',
        unsafe_allow_html=True,
    )

    # --- Render Confidence ---
    conf_text = content_blocks.get("5. CONFIDENCE LEVEL", "")
    st.markdown(f"""
    <div class="premium-card">
        <div class="card-header">🛡️ Confidence Level</div>
        <div style="font-size: 1rem; color: var(--text-color);">
            {conf_text.replace('\n', '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Render Emergency Resources ---
    res_text = content_blocks.get("6. RELEVANT EMERGENCY RESOURCES", "")

    if res_text and "no direct" not in res_text.lower():

        raw_html = res_text.replace("\n", "<br>")

        def _linkify_phone(match):
            prefix, number = match.group(1), match.group(2)
            tel_number = re.sub(r"[\s\-]", "", number)
            return f'{prefix}<a href="tel:{tel_number}">{number}</a>'

        def _linkify_segment(segment):
            # Bare domains/portals without a scheme, e.g. "ceir.gov.in"
            segment = re.sub(
                r'\b((?:[a-zA-Z0-9-]+\.)+(?:gov\.in|nic\.in|co\.in|org\.in|com|org|net|in))\b',
                r'<a href="https://\1" target="_blank">\1</a>',
                segment
            )
            # Phone/helpline numbers, e.g. "Police Emergency: 112"
            segment = re.sub(
                r'(:\s*)(\d[\d\s\-]{1,14}\d)(?!\d)',
                _linkify_phone,
                segment
            )
            return segment

        # Process full http(s) URLs first, then only linkify the remaining
        # plain-text pieces so we never re-wrap something already inside <a>...</a>
        pieces = re.split(r'(https?://[^\s<]+)', raw_html)
        for i, piece in enumerate(pieces):
            if piece.startswith("http://") or piece.startswith("https://"):
                pieces[i] = f'<a href="{piece}" target="_blank">{piece}</a>'
            else:
                pieces[i] = _linkify_segment(piece)
        res_html = "".join(pieces)

        st.markdown(f"""
        <div class="premium-card">
            <div class="card-header">🚨 Relevant Emergency Resources</div>
            <div style="font-size: 0.9rem; color: var(--text-color); margin-bottom: 0.8rem;">
                {res_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Render Disclaimer ---
    disc_text = content_blocks.get("7. DISCLAIMER", "")
    if not disc_text:
        disc_text = "Disclaimer: This is an AI-powered assistant designed for informational purposes. It is not a substitute for professional legal counsel. Please consult a qualified lawyer for official legal advice."
    st.markdown(f"""
    <div class="disclaimer-alert">
        <div style="font-size: 1.2rem; line-height: 1;">⚠️</div>
        <div>{disc_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Expandable retrieved text section ---
    st.write("")
    with st.expander("🔍 View Raw Legal Text Passages (FAISS Grounding Source)"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"""
            <div class="source-block">
                <div class="source-meta">
                    <strong>Document:</strong> {doc['source']} | 
                    <strong>Page(s):</strong> {", ".join(map(str, doc['pages']))} | 
                    <strong>Cosine Similarity:</strong> {doc['score']:.4f}
                </div>
                <div style="font-size: 0.9rem; line-height: 1.5; font-style: italic; color: #D1D5DB;">
                    "{doc['text']}"
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- Expandable web search grounding references ---
    if grounding_metadata:
        web_queries = []
        web_sources = []
        
        # 1. Extract queries (attribute & dictionary fallbacks)
        if hasattr(grounding_metadata, "web_search_queries") and grounding_metadata.web_search_queries:
            web_queries = list(grounding_metadata.web_search_queries)
        elif isinstance(grounding_metadata, dict) and "web_search_queries" in grounding_metadata:
            web_queries = list(grounding_metadata["web_search_queries"])
        elif isinstance(grounding_metadata, dict) and "webSearchQueries" in grounding_metadata:
            web_queries = list(grounding_metadata["webSearchQueries"])
            
        # 2. Extract chunks (attribute & dictionary fallbacks)
        chunks = []
        if hasattr(grounding_metadata, "grounding_chunks") and grounding_metadata.grounding_chunks:
            chunks = grounding_metadata.grounding_chunks
        elif isinstance(grounding_metadata, dict) and "grounding_chunks" in grounding_metadata:
            chunks = grounding_metadata["grounding_chunks"]
        elif isinstance(grounding_metadata, dict) and "groundingChunks" in grounding_metadata:
            chunks = grounding_metadata["groundingChunks"]
            
        if chunks:
            seen_urls = set()
            for chunk in chunks:
                web_data = None
                if hasattr(chunk, "web") and chunk.web:
                    web_data = chunk.web
                elif isinstance(chunk, dict) and "web" in chunk:
                    web_data = chunk["web"]
                    
                if web_data:
                    title = "Web Result"
                    url = None
                    
                    if hasattr(web_data, "title"):
                        title = getattr(web_data, "title", "Web Result")
                    elif isinstance(web_data, dict) and "title" in web_data:
                        title = web_data["title"]
                        
                    if hasattr(web_data, "uri"):
                        url = getattr(web_data, "uri", None)
                    elif isinstance(web_data, dict) and "uri" in web_data:
                        url = web_data["uri"]
                        
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        web_sources.append({"title": title, "url": url})
                        
        if web_sources:
            with st.expander("🌐 View Live Web Search References (Google Grounding Source)"):
                if web_queries:
                    st.markdown(f"**Search Queries Executed:** `{', '.join(web_queries)}`")
                for src in web_sources:
                    st.markdown(f"""
                    <div style="background-color: #1E293B; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.6rem; border-left: 3px solid #FFB703;">
                        <a href="{src['url']}" target="_blank" style="text-decoration: none; color: #8ecae6; font-weight: bold;">
                            🔗 {src['title']}
                        </a>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem;">
                            URL: {src['url']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Main Application Flow
def main():
    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h2 style="color: #FFB703; margin-bottom: 0.2rem;">⚖️ NyayaMitra AI</h2>
        <span style="font-size: 0.85rem; color: #8A9CA8;">Legal Decision Intelligence</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.info(
        "NyayaMitra is designed to empower Indian citizens by translating complex laws into "
        "simple, actionable information in their own language."
    )
    
    st.sidebar.subheader("Configuration Status")
    
    # Check variables
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    if use_vertex:
        st.sidebar.success("Vertex AI Mode Active")
        st.sidebar.markdown(f"**Project:** `{os.getenv('GOOGLE_CLOUD_PROJECT', 'unset')}`")
        st.sidebar.markdown(f"**Location:** `{os.getenv('GOOGLE_CLOUD_LOCATION', 'unset')}`")
    else:
        st.sidebar.success("Google AI Studio Mode Active")
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Loaded Legal Docs")
    st.sidebar.markdown("- Constitution of India")
    st.sidebar.markdown("- Bharatiya Nyaya Sanhita (BNS)")
    st.sidebar.markdown("- Bharatiya Nagarik Suraksha Sanhita (BNSS)")
    st.sidebar.markdown("- Bharatiya Sakshya Adhiniyam (BSA)")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Web Search Grounding")
    enable_web_search = st.sidebar.checkbox(
        "🌐 Enable Google Search Supplement", 
        value=True, 
        help="Automatically searches Google Search for live legal updates and fallback citations when local vector store similarity scores are low."
    )
    
    # Main content header
    st.markdown('<div class="main-title">NyayaMitra AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Decision Intelligence for Citizen Empowerment & Legal Literacy</div>', unsafe_allow_html=True)
    
    # Load resources
    index, metadata = load_vector_store()
    
    if index is None or metadata is None:
        st.warning("⚠️ Local Vector Store is not initialized.")
        st.info("Please run the offline indexing script `python build_index.py` first to process the legal documents and generate the FAISS index.")
        return
        
    client = get_genai_client()
    
    # Input container
    st.markdown("### 📥 Describe Your Legal Situation")
    
    # Create two columns for text description vs file upload
    col1, col2 = st.columns([3, 2])
    
    with col1:
        user_text = st.text_area(
            "Describe your problem in plain language (Hindi, Telugu, English, etc.):",
            placeholder="Type your query here. E.g., 'What are my rights if my landlord threatens to evict me without notice?' or 'मेरी दुकान पर किसी ने चोरी की, मुझे FIR कैसे दर्ज करनी चाहिए?'",
            height=200
        )
        
    with col2:
        uploaded_file = st.file_uploader(
            "Or upload a photo/PDF of a legal notice, FIR, or letter:",
            type=["png", "jpg", "jpeg", "pdf"]
        )
        
    submit_button = st.button("⚖️ Get Legal Decision Support", use_container_width=True)
    
    if submit_button:
        if not user_text and not uploaded_file:
            st.error("Please describe your issue in the text area or upload a document.")
            return
            
        # Pipeline execution
        st.markdown("---")
        
        # 1. Multimodal document transcription if present
        query_to_search = user_text
        extracted_text_summary = None
        
        if uploaded_file:
            extracted_text_summary = process_multimodal_file(client, uploaded_file)
            if extracted_text_summary:
                st.info("📄 **Document understanding result:**")
                st.write(extracted_text_summary)
                
                # Combine user typed issue with document summary
                if user_text:
                    query_to_search = f"{user_text}\n\n[Document Context]: {extracted_text_summary}"
                else:
                    query_to_search = extracted_text_summary
                    
        if not query_to_search:
            return
            
        # 2. Retrieve relevant passages
        with st.spinner("Retrieving relevant articles and sections from Indian legal texts..."):
            try:
                retrieved_docs = retrieve_relevant_documents(client, index, metadata, query_to_search, k=5)
            except Exception as e:
                st.error(f"Error during vector search: {e}")
                return
                
        # 3. Generate answers
        with st.spinner("Analyzing laws and preparing decision intelligence plan..."):
            try:
                # Constrain search triggering based on FAISS similarity scores
                max_score = max([doc["score"] for doc in retrieved_docs]) if retrieved_docs else 0.0
                should_trigger_search = enable_web_search and (max_score < 0.65)
                
                if should_trigger_search:
                    st.info("🔍 *Local grounding similarity is low/weak. Activating live Google Search fallback grounding...*")
                else:
                    if enable_web_search:
                        st.success(f"✅ *Local legal store grounding is strong (Similarity: {max_score:.4f}). Google Search bypassed.*")
                        
                answer, grounding_metadata = generate_legal_answer(
                    client, query_to_search, retrieved_docs, enable_web_search=should_trigger_search
                )
            except Exception as e:
                st.error(f"Error during generative pipeline: {e}")
                return
                
        # 4. Display results using premium styled HTML cards
        st.markdown("### 📋 Decision Intelligence Report")
        parse_markdown_to_premium_ui(answer, retrieved_docs, grounding_metadata)

if __name__ == "__main__":
    main()
