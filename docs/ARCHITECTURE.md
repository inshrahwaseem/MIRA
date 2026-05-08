# System Architecture - MIRA

## 1. High-Level ASCII Diagram

```text
[ FRONTEND - Next.js 14 ]
      |
      | (SSE / WebSockets)
      v
[ API GATEWAY - Express.js ] <------> [ AUTH - Clerk ]
      |
      +-----> [ DATABASE - Supabase (Postgres + pgvector) ]
      |
      +-----> [ CACHE/LIMIT - Upstash Redis ]
      |
      +-----> [ ML ORCHESTRATOR - FastAPI ]
                |
                +---> [ TEXT CHANNEL: DistilRoBERTa (HF) ]
                |
                +---> [ VOICE CHANNEL: Librosa + Whisper (Local) ]
                |
                +---> [ CAMERA CHANNEL: MediaPipe + DeepFace ]
                |
                +---> [ FUSION ENGINE: Weighted Bayesian ]
                |
                +---> [ LLM ENGINE: Ollama (Llama 3) ]
                |       |
                |       +---> [ RAG: ChromaDB + LlamaIndex ]
                |
                +---> [ ANALYTICS: MLflow + Langfuse ]
```

## 2. Module Responsibilities

### Frontend (Next.js 14)
*   **UI/UX:** Responsive 3-column layout, Emotion Orb (Framer Motion), Voice Waveforms.
*   **Sensors:** MediaPipe browser-side for face/pose landmarks (reducing payload size).
*   **Visualization:** Recharts for 2D history, Plotly/Three.js for 3D Emotion Universe.

### API Gateway (Express.js)
*   **Routing:** Handles session management, user profiles, and activity logging.
*   **Streaming:** Acts as a proxy for SSE streams from the ML service to the frontend.
*   **Security:** Enforces Clerk auth and Upstash rate limits.

### ML Service (FastAPI)
*   **Processing:** Heavy lifting for Text/Voice/Camera feature extraction.
*   **Fusion:** Implements the Weighted Bayesian Fusion to resolve modal conflicts.
*   **Psychology Engine:** Logic for CBT distortion detection and Attachment Style profiling.
*   **Local LLM:** Orchestrates Ollama requests via LangChain.

### Persistence & Memory
*   **Supabase:** Primary storage for user metadata, session logs, and long-term mood vectors.
*   **ChromaDB:** Local vector store for the RAG knowledge base (Psychology textbooks).
*   **pgvector:** Enables semantic search within the main SQL database.

## 3. Data Flow

1.  **Ingestion:** Frontend captures text, audio chunks, and facial landmarks (MediaPipe).
2.  **Preprocessing:** FastAPI receives payload; extracts MFCCs (voice), classify action units (camera), and tokenizes text.
3.  **Analysis:**
    *   Text classified into 40 Plutchik emotions.
    *   Voice analyzed for pitch/energy/temporal markers.
    *   Camera maps Action Units to Ekman micro-expressions.
4.  **Fusion:** The "Weighted Bayesian Fusion" combines outputs. If Text="Happy" but Voice="Anxious" (high energy/jitter), the result leans toward "Masked Anxiety."
5.  **Inference:** Fused result + RAG context (CBT/ACT) sent to Ollama.
6.  **Response:** Llama 3 generates empathetic, theory-grounded response, streamed via SSE.
7.  **Tracking:** ML metrics sent to MLflow; LLM traces sent to Langfuse.

## 4. Technology Justifications

| Tech | Reason |
| :--- | :--- |
| **Next.js 14** | Best-in-class SSR/ISR, App Router for complex layouts, TypeScript integration. |
| **FastAPI** | High performance for Python ML tasks, Pydantic validation, native async support. |
| **Ollama (Llama 3)** | Privacy-first; ensures no sensitive mental health data leaves the local environment. |
| **MediaPipe** | Extremely efficient browser-side landmark detection (468 facial points). |
| **ChromaDB** | Lightweight, local vector store perfect for RAG without external dependencies. |
| **Supabase** | Rapid development with built-in Auth (via Clerk integration), DB, and Vector support. |
| **Librosa** | Industry standard for MFCC and acoustic feature extraction. |
| **Framer Motion** | Critical for the "Bioluminescent Sanctuary" aesthetic (organic transitions). |
| **Langfuse** | Essential for observability into LLM prompts and evaluation. |
