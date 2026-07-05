# NyayaMitra AI — Legal Decision Intelligence Platform

NyayaMitra AI is a Legal Decision Intelligence Platform built as a working prototype for the Google Gen AI Academy Hackathon. Under the theme of **"AI for Better Living and Smarter Communities"**, this Citizen Engagement / Accessibility solution helps common Indian citizens understand their legal rights and outlines clear step-by-step action plans in simple, plain, non-legal language.

---

## Key Features

1.  **Issue Classification**: Automatically categorizes the citizen's legal problem (Criminal, Civil, Cyber Crime, Family Dispute, etc.).
2.  **Strict RAG Grounding**: Retrieves matching sections from primary Indian legal texts (Constitution, BNS, BNSS, BSA). Cites exact section/article numbers *only* if they exist in the retrieved database (no hallucinated citations).
3.  **Plain Language Explanation**: Translates dense legal code into simple, non-legal language.
4.  **Step-by-Step Action Plan**: Provides actionable steps (how to reply, where to file, what portals to visit).
5.  **Grounded Confidence Indicator**: Displays a clear rating (High / Medium / Low Confidence) based directly on the FAISS cosine similarity scores of the retrieved context.
6.  **Emergency Helplines**: Surfaces specific citizen support helplines (Police: 112, Women Helpline: 1091, Cyber Crime: 1930, National Consumer Helpline: 1915) when relevant to the issue category.
7.  **Native Multimodal Input**: Supports text descriptions *or* direct image/PDF document uploads of notices, letters, or FIRs. Gemini reads the document natively and passes it to the RAG retrieval pipeline.
8.  **Native Language Matching**: Automatically responds in the same language the user wrote in (Hindi, Telugu, English, etc.) without intermediate translation steps.

---

## Project Structure

```
nyayamitra/
├── .env.example        # Environment variables template
├── .env                # Active configuration (ignored by git)
├── app.py              # Streamlit UI & Retrieval Pipeline
├── build_index.py      # Precomputation script to build FAISS store
├── requirements.txt    # Python library dependencies
├── Dockerfile          # Container configuration for Cloud Run
├── README.md           # This documentation
├── legal_docs/         # PDF source documents (Constitution, BNS, BNSS, BSA)
└── vector_store/       # Precomputed FAISS index & metadata files
```

---

## Environment Configuration

NyayaMitra utilizes Google's modern unified Python SDK (`google-genai`). The client dynamically configures itself to run on **Google AI Studio (API Key)** or **Google Cloud Vertex AI (GCP Project)** based purely on environment variables, requiring no code modifications.

Create an active `.env` file in the `nyayamitra/` directory:

```bash
# --- GOOGLE AI STUDIO (Gemini API Key) ---
# Set this to use Gemini Developer API / Google AI Studio.
GEMINI_API_KEY=your_gemini_api_key_here

# --- GOOGLE VERTEX AI (GCP) ---
# To switch to Vertex AI (GCP) mode, set GOOGLE_GENAI_USE_VERTEXAI to true.
# In this mode, the SDK ignores GEMINI_API_KEY and uses Google Cloud credentials.
GOOGLE_GENAI_USE_VERTEXAI=false
GOOGLE_CLOUD_PROJECT=your_gcp_project_id_here
GOOGLE_CLOUD_LOCATION=us-central1

# Optional: Path to GCP Service Account JSON key if testing Vertex AI locally
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

To swap from AI Studio to Vertex AI, simply toggle `GOOGLE_GENAI_USE_VERTEXAI=true` and provide your GCP project details.

---

## Local Setup & Run

### 1. Installation
Ensure you are using Python 3.10+ (tested on Python 3.14 on Windows). Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate FAISS Vector Index (Precomputation)
Place your legal source PDFs (e.g. `BNS_2023.pdf`, `BNSS_2023.pdf`, `BNA_2023.pdf` / `BSA_2023.pdf`, `constitution.pdf`) inside the `legal_docs/` folder.

Run the offline indexing script:
```bash
python build_index.py
```
*   **Chunking Strategy**: The script parses PDFs, concatenating pages first and mapping page transition indices as character metadata, which prevents cutting a legal clause in half at page boundaries. It chunks the text recursively (6,000 characters with 1,000 character overlap) to preserve dense legal context and keep the chunk count (~470) within free-tier API daily limits.
*   **Embeddings**: Embeds chunks using Google's `gemini-embedding-001` (768 dimensions) in batches with exponential backoff retries to manage rate limits.
*   **Outputs**: Saves the vector search index to `vector_store/index.faiss` and serialized text metadata to `vector_store/metadata.json`.

### 3. Launch Streamlit Application
Start the local server:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Deployment Options

### Path A: Streamlit Community Cloud (Zero-Ops)
Streamlit Community Cloud hosts directly from your GitHub repository:
1.  Commit the files including the prebuilt `vector_store/` directory.
2.  Deploy the app on Streamlit Cloud using the repository branch pointing to `nyayamitra/app.py`.
3.  Go to the app settings dashboard, navigate to **Secrets**, and enter your AI Studio key:
    ```toml
    GEMINI_API_KEY = "your_gemini_api_key_here"
    ```
Streamlit Cloud automatically reads the `requirements.txt` file and handles dependencies.

### Path B: Google Cloud Run (Containerized)
To deploy as a managed container on GCP Cloud Run, hand these commands to your teammate:

1.  **Configure environment variables**: Make sure your local terminal has GCP authenticated:
    ```bash
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID
    ```

2.  **Enable necessary GCP services**:
    ```bash
    gcloud services enable artifactregistry.googleapis.com run.googleapis.com
    ```

3.  **Create an Artifact Registry repository**:
    ```bash
    gcloud artifacts repositories create nyayamitra-repo \
        --repository-format=docker \
        --location=us-central1 \
        --description="Docker repository for NyayaMitra AI"
    ```

4.  **Build and push the Docker image to Artifact Registry**:
    ```bash
    gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/nyayamitra-repo/nyayamitra-app:latest .
    ```

5.  **Deploy to Cloud Run**:
    Pass the API key configuration as environment variables directly to the container:
    ```bash
    gcloud run deploy nyayamitra-service \
        --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/nyayamitra-repo/nyayamitra-app:latest \
        --region=us-central1 \
        --allow-unauthenticated \
        --set-env-vars="GEMINI_API_KEY=YOUR_GEMINI_API_KEY" \
        --port=8080
    ```
    *Note: If deploying under GCP Vertex AI, set `--set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1"` instead, and ensure the Cloud Run service account has the **Vertex AI User** IAM role.*
