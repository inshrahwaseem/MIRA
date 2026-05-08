PROJECT: MIRA — Multimodal Mental Health Companion AI
DESCRIPTION: AI psychiatrist that analyzes emotion via text + voice + camera simultaneously

TECH STACK (never deviate without asking):
- Frontend: Next.js 14 (App Router) + TypeScript strict + Tailwind CSS + shadcn/ui customized
- Backend Node.js: Express.js + TypeScript (API gateway, streaming via SSE, webhooks)
- Backend Python: FastAPI + Pydantic v2 (for ALL ML/AI models — never put ML in Node)
- Database Primary: Supabase PostgreSQL + pgvector (for vector similarity search)
- Database Vector: ChromaDB local (for RAG knowledge base + long-term memory)
- LLM Local: Ollama (llama3 model) — privacy-first, no data leaves device
- LLM Cloud Fallback: Groq API (free, ultra-fast) → Anthropic Haiku (cheap)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace, free, local, no API key)
- Auth: Clerk (never build auth from scratch)
- Email: Resend (resend.com)
- Automation: n8n (self-hosted via Docker, free)
- Monitoring LLM: Langfuse (self-hosted, free)
- Error Tracking: Sentry (free tier)
- Experiment Tracking: MLflow (self-hosted, free)
- Rate Limiting: Upstash Redis

CODING CONVENTIONS:
- Python: snake_case, type hints everywhere, Pydantic for all data models, no global variables
- TypeScript: strict mode, no 'any' types, interfaces for all data shapes
- Functions: max 50 lines, single responsibility
- Variables: descriptive names only (no x, temp, data, result)
- All async/await — no .then() chains
- Docstrings on every function (Python) / JSDoc on every export (TypeScript)

SECURITY RULES:
- API keys ONLY in .env files — never hardcode, never frontend
- Rate limiting on ALL API endpoints (Upstash Redis)
- Input validation: Zod (TypeScript) + Pydantic (Python) — BOTH frontend AND backend
- Prompt injection prevention: system prompt and user input always separate
- Never return stack traces in production error responses

TESTING REQUIREMENTS:
- Minimum coverage: 85%
- Python: pytest + pytest-asyncio
- TypeScript: Vitest + React Testing Library
- E2E: Playwright (5 critical user flows)
- Tests written alongside code, not after

DEPLOYMENT:
- Frontend: Vercel
- Backend Node.js: Railway
- Python ML Service: Railway (separate service)
- n8n Automation: Railway (Docker container)
- Never use SQLite in production — always Supabase PostgreSQL
