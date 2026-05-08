<div align="center">
  <img src="https://raw.githubusercontent.com/inshrahwaseem/MIRA/main/docs/assets/mira_logo.png" alt="MIRA Logo" width="200" />
  
  # MIRA — Multimodal Mental Health Companion AI

  **Clinical-grade emotion fusion. Privacy-first architecture. Truly empathetic AI.**

  [![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![LangChain](https://img.shields.io/badge/LangChain-ReAct-black?logo=langchain)](https://langchain.com/)
  [![Supabase](https://img.shields.io/badge/Supabase-DB-3ECF8E?logo=supabase)](https://supabase.com/)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

<br/>

MIRA (Multimodal Intelligent Response Agent) is a next-generation psychological AI companion. By seamlessly fusing **Text, Voice, and Facial Expressions** in real-time using Bayesian cross-modal analysis, MIRA detects genuine emotional states, even when users mask their feelings.

Driven by a local Llama 3 engine via the ReAct framework, MIRA provides clinically grounded, safe, and personalized therapeutic interventions (like CBT and ACT) while strictly adhering to privacy-first edge computing paradigms.

---

## ✨ Core Features

- 👁️ **Multimodal Fusion Engine:** Paralleled text sentiment, vocal prosody (pitch/energy), and facial micro-expressions combined via Bayesian weighting.
- 🛡️ **Cross-Modal Conflict Detection:** Identifies suppressed anxiety or masked positivity (e.g., smiling but speaking with a hopeless tone).
- 🧠 **LangChain ReAct Agent:** Empowered with 7 distinct psychological tools, including long-term vector memory lookup and therapeutic knowledge retrieval.
- 🎨 **Dynamic Atmosphere:** The UI continuously adapts its color palette, animations, and typography based on the user's real-time emotional valence and arousal.
- 🔒 **Privacy-First:** "Off-the-grid" capability using local Ollama and ChromaDB. No camera data is ever recorded or transmitted.

---

## 🏗️ Architecture

MIRA operates on a sophisticated 9-layer security and data architecture:

1. **Web App (Next.js 14):** Fluid, glassmorphic UI using Framer Motion and custom CSS properties.
2. **API Gateway (Express.js):** Rate limiting, input validation (Zod), and auth routing.
3. **ML Service (FastAPI):** Orchestrates the `MultimodalFusionEngine` using `asyncio.gather` for parallel real-time analysis.
4. **LLM Service (FastAPI):** LangChain AgentExecutor utilizing RAG (Retrieval-Augmented Generation) against psychology textbooks.

*(See detailed diagrams in `/docs`)*

---

## 💻 Tech Stack

### Frontend
- **Framework:** Next.js 14 (App Router), React 18
- **Styling:** Vanilla CSS Custom Properties, Tailwind CSS
- **Animation:** Framer Motion
- **Visualization:** Recharts, Plotly

### Backend
- **Microservices:** FastAPI (Python), Express.js (TypeScript)
- **AI/ML Engine:** LangChain, Scikit-learn, OpenCV (Face), Librosa (Audio)
- **Local LLM:** Ollama (Llama 3 8B Instruct)
- **Database:** Supabase (PostgreSQL), ChromaDB (Vector)

---

## 🚀 Quick Start

### Prerequisites
- Node.js (v20+)
- Python (v3.11+)
- Docker & Docker Compose
- Ollama (installed locally with `llama3` pulled)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/inshrahwaseem/MIRA.git
   cd MIRA
   ```

2. **Environment Variables:**
   Copy the example environment file and fill in your Supabase and API credentials:
   ```bash
   cp .env.example .env.local
   ```

3. **Install Dependencies & Start:**
   Using the built-in Makefile:
   ```bash
   make setup
   make dev
   ```

   *Alternatively, using Docker Compose:*
   ```bash
   docker-compose up --build
   ```

4. **Access the Application:**
   Navigate to `http://localhost:3000` in your browser.

---

## 📁 Repository Structure

```text
MIRA/
├── apps/
│   ├── web/               # Next.js 14 Frontend UI
│   └── api-gateway/       # Express Router & Security
├── services/
│   ├── ml-service/        # Multimodal Fusion & Emotion Taxonomy
│   └── llm-service/       # LangChain ReAct Agent & RAG Memory
├── packages/
│   └── shared-types/      # Zod validation and TS Interfaces
├── database/              # Supabase Migrations
└── docs/                  # PRDs, API Contracts, Architecture
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/inshrahwaseem/MIRA/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
  <p>Built with ❤️ by <a href="https://github.com/inshrahwaseem">Inshrah Waseem</a></p>
  <p><i>Disclaimer: MIRA is an AI research project and not a licensed medical professional or crisis service.</i></p>
</div>
