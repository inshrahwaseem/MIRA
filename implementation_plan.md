# MIRA Monorepo Scaffold Implementation Plan

This plan details the complete folder structure and all critical configuration files required to build the production monorepo for MIRA, following the approved Architecture and PRD.

## User Review Required
Please review the complete file tree and the provided configuration files below. Once approved, I will automatically generate all the files on your filesystem.

---

## 1. Full File Tree

```text
mira/
├── apps/
│   ├── web/                          ← Next.js 14 frontend
│   │   ├── app/
│   │   │   ├── (auth)/
│   │   │   │   ├── login/page.tsx
│   │   │   │   └── register/page.tsx
│   │   │   ├── dashboard/
│   │   │   │   ├── page.tsx
│   │   │   │   ├── layout.tsx
│   │   │   │   ├── check-in/page.tsx
│   │   │   │   ├── mood-map/page.tsx
│   │   │   │   └── insights/page.tsx
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   ├── shared/
│   │   │   └── features/
│   │   ├── lib/
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── next.config.mjs
│   │   ├── tailwind.config.ts
│   │   └── Dockerfile
│   └── api-gateway/                  ← Express.js Node.js backend
│       ├── src/
│       │   ├── controllers/
│       │   ├── middleware/
│       │   ├── routes/
│       │   └── index.ts
│       ├── package.json
│       ├── tsconfig.json
│       └── Dockerfile
├── services/
│   ├── ml-service/                   ← FastAPI Python — all ML models
│   │   ├── app/
│   │   │   ├── core/
│   │   │   ├── models/
│   │   │   ├── routers/
│   │   │   ├── services/
│   │   │   │   ├── text/
│   │   │   │   ├── voice/
│   │   │   │   ├── camera/
│   │   │   │   └── fusion/
│   │   │   ├── utils/
│   │   │   └── main.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── llm-service/                  ← FastAPI Python — LLM + RAG + Agent
│   │   ├── app/
│   │   │   ├── rag/
│   │   │   ├── memory/
│   │   │   ├── agent/
│   │   │   ├── psychology/
│   │   │   ├── companion/
│   │   │   └── main.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── automation/                   ← n8n config (managed via docker-compose)
├── packages/
│   ├── shared-types/                 ← TypeScript interfaces shared across apps
│   │   ├── src/
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── shared-utils/                 ← Shared utilities
│       ├── src/
│       │   ├── index.ts
│       │   ├── env.ts
│       │   └── logger.ts
│       ├── package.json
│       └── tsconfig.json
├── docker-compose.yml                ← Local dev: ALL services running together
├── docker-compose.prod.yml           ← Production config
├── .env.example                      ← All required vars listed, NO values
├── Makefile                          ← make setup, make dev, make test, make deploy
├── turbo.json                        ← Turborepo monorepo pipeline config
├── package.json                      ← Monorepo root package.json
└── README.md
```

---

## 2. Root Configuration Files

### `turbo.json`
```json
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"]
    },
    "lint": {
      "dependsOn": ["^lint"]
    },
    "test": {
      "dependsOn": ["^build"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

### `Makefile`
```makefile
.PHONY: setup dev test lint deploy db-migrate

setup:
	pnpm install
	cd services/ml-service && pip install -r requirements.txt
	cd services/llm-service && pip install -r requirements.txt

dev:
	docker-compose up --build

test:
	pnpm run test
	cd services/ml-service && pytest
	cd services/llm-service && pytest

lint:
	pnpm run lint
	cd services/ml-service && ruff check . && mypy .
	cd services/llm-service && ruff check . && mypy .

deploy:
	vercel deploy --prod
	railway up

db-migrate:
	pnpm --filter api-gateway run migrate
```

### `.env.example`
```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
PYTHON_ML_SERVICE_URL=http://ml-service:8000
PYTHON_LLM_SERVICE_URL=http://llm-service:8001
OLLAMA_API_URL=http://ollama:11434
OLLAMA_MODEL=llama3
GROQ_API_KEY=
RESEND_API_KEY=
UPSTASH_REDIS_URL=
UPSTASH_REDIS_TOKEN=
INTERNAL_API_SECRET=
MLFLOW_TRACKING_URI=http://mlflow:5000
LANGFUSE_HOST=http://langfuse:3002
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
SENTRY_DSN=
NEXT_PUBLIC_APP_URL=
```

---

## 3. Docker Configurations

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: apps/web/Dockerfile
    ports:
      - "3000:3000"
    env_file:
      - .env
    depends_on:
      - api-gateway

  api-gateway:
    build:
      context: .
      dockerfile: apps/api-gateway/Dockerfile
    ports:
      - "4000:4000"
    env_file:
      - .env
    depends_on:
      - ml-service
      - llm-service

  ml-service:
    build:
      context: ./services/ml-service
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./services/ml-service:/app

  llm-service:
    build:
      context: ./services/llm-service
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    env_file:
      - .env
    volumes:
      - ./services/llm-service:/app
    depends_on:
      - ollama
      - chromadb

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  chromadb:
    image: chromadb/chroma:0.4.22
    ports:
      - "8002:8000"
    volumes:
      - chromadb_data:/chroma/chroma

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000

  langfuse:
    image: langfuse/langfuse:2.7.0
    ports:
      - "3002:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/langfuse
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    ports:
      - "5432:5432"

  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    volumes:
      - n8n_data:/home/node/.n8n

volumes:
  ollama_data:
  chromadb_data:
  n8n_data:
```

### Python Service Dockerfile (`services/ml-service/Dockerfile` & `services/llm-service/Dockerfile`)
*(Same baseline format for both services, ensuring non-root and healthchecks)*
```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies (e.g. for OpenCV/dlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Node Service Dockerfile (`apps/api-gateway/Dockerfile`)
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install
COPY . .
RUN pnpm run build

FROM node:20-alpine AS runner
WORKDIR /app
RUN addgroup -S nodeapp && adduser -S nodeapp -G nodeapp
COPY --from=builder /app/package.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist

USER nodeapp

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:4000/health || exit 1

EXPOSE 4000
CMD ["node", "dist/index.js"]
```

---

## 4. Service Dependencies

### `services/ml-service/requirements.txt` & `services/llm-service/requirements.txt`
```text
transformers==4.38.0
torch==2.2.0
mediapipe==0.10.9
opencv-python==4.9.0.80
librosa==0.10.1
scikit-learn==1.4.0
langchain==0.1.14
chromadb==0.4.22
ollama==0.1.7
fastapi==0.109.0
uvicorn==0.27.0
pandas==2.2.0
numpy==1.26.4
openai-whisper==20231117
dlib==19.24.2
deepface==0.0.89
fer==22.5.1
umap-learn==0.5.5
pydantic==2.6.0
mlxtend==0.23.0
mlflow==2.10.0
langfuse==2.7.0
sentence-transformers==2.4.0
llama-index==0.10.0
scipy==1.12.0
sounddevice==0.4.6
shap==0.44.0
```

### `apps/web/package.json`
```json
{
  "name": "web",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "framer-motion": "11.0.8",
    "recharts": "2.12.2",
    "three": "0.162.0",
    "@clerk/nextjs": "4.29.9",
    "lucide-react": "0.344.0",
    "shared-types": "workspace:*",
    "shared-utils": "workspace:*"
  },
  "devDependencies": {
    "@types/node": "20.11.24",
    "@types/react": "18.2.61",
    "@types/react-dom": "18.2.19",
    "typescript": "5.3.3",
    "tailwindcss": "3.4.1",
    "eslint": "8.57.0",
    "eslint-config-next": "14.1.0"
  }
}
```

### `apps/api-gateway/package.json`
```json
{
  "name": "api-gateway",
  "version": "1.0.0",
  "scripts": {
    "dev": "ts-node-dev src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "lint": "eslint src/**/*.ts"
  },
  "dependencies": {
    "express": "4.18.3",
    "cors": "2.8.5",
    "helmet": "7.1.0",
    "@clerk/clerk-sdk-node": "4.13.11",
    "shared-types": "workspace:*",
    "shared-utils": "workspace:*"
  },
  "devDependencies": {
    "@types/express": "4.17.21",
    "@types/cors": "2.8.17",
    "typescript": "5.3.3",
    "ts-node-dev": "2.0.0"
  }
}
```

---

## 5. TypeScript Shared Interfaces (`packages/shared-types/src/index.ts`)

```typescript
export enum EmotionName {
  Joy = "Joy", Serenity = "Serenity", Ecstasy = "Ecstasy",
  Trust = "Trust", Acceptance = "Acceptance", Admiration = "Admiration",
  Fear = "Fear", Apprehension = "Apprehension", Terror = "Terror",
  Surprise = "Surprise", Distraction = "Distraction", Amazement = "Amazement",
  Sadness = "Sadness", Pensiveness = "Pensiveness", Grief = "Grief",
  Disgust = "Disgust", Boredom = "Boredom", Loathing = "Loathing",
  Anger = "Anger", Annoyance = "Annoyance", Rage = "Rage",
  Anticipation = "Anticipation", Interest = "Interest", Vigilance = "Vigilance",
  // Complex Dyads & Social mapping...
}

export interface EmotionScore {
  emotion: EmotionName;
  intensity: number;
}

export interface EmotionVector {
  emotion: EmotionName;
  intensity: number;
  valence: number;
  arousal: number;
  dominance: number;
}

export interface TextAnalysisResult {
  emotions: EmotionScore[];
  sentiment: number;
}

export interface VoiceFeatures {
  pitch: { mean: number; std: number; range: number; slope: number; voicedFraction: number };
  energy: { rmsMean: number; rmsStd: number; loudness: number };
  temporal: { speechRate: number; pauseCount: number; pauseDuration: number; articulationRate: number };
  quality: { jitter: number; shimmer: number; hnr: number };
}

export interface VoiceAnalysisResult {
  features: VoiceFeatures;
  emotions: EmotionScore[];
}

export interface FaceResult {
  actionUnits: Record<string, number>;
  microExpressions: string[];
  smileType: 'genuine' | 'polite' | 'none';
}

export interface GazeResult {
  eyeContactPercentage: number;
  blinkRate: number;
  direction: string;
}

export interface PostureResult {
  shoulderSlope: number;
  headTilt: number;
  slouchingSeverity: number;
  openness: number;
}

export interface HandResult {
  fidgetingScore: number;
  selfTouchingFrequency: number;
  tension: number;
}

export interface CameraAnalysisResult {
  face: FaceResult;
  gaze: GazeResult;
  posture: PostureResult;
  hands: HandResult;
}

export interface ConflictReport {
  isConflict: boolean;
  description?: string;
}

export interface FusedEmotionResult {
  primaryEmotion: EmotionName;
  secondaryEmotion?: EmotionName;
  intensityLabel: 'mild' | 'moderate' | 'intense';
  conflictReport: ConflictReport;
  vectors: EmotionVector[];
}

export interface DriftReport {
  trend: 'improving' | 'declining' | 'stable';
  variance: number;
}

export interface TriggerRule {
  antecedents: string[];
  consequents: string[];
  confidence: number;
}

export enum AttachmentStyle {
  Secure = "Secure",
  Anxious = "Anxious",
  Avoidant = "Avoidant",
  Fearful = "Fearful"
}

export interface BigFiveScores {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

export interface UserProfile {
  id: string;
  attachmentStyle: AttachmentStyle;
  bigFive: BigFiveScores;
}

export interface SessionData {
  sessionId: string;
  userId: string;
  timestamp: string;
  fusedEmotion: FusedEmotionResult;
  transcript: ChatMessage[];
}

export enum TherapyTheory { CBT = "CBT", ACT = "ACT", General = "General" }

export enum CognitivDistortionType {
  Catastrophizing = "Catastrophizing",
  BlackAndWhiteThinking = "BlackAndWhiteThinking",
  MindReading = "MindReading",
  FortuneTelling = "FortuneTelling",
  Personalization = "Personalization",
  EmotionalReasoning = "EmotionalReasoning",
  ShouldStatements = "ShouldStatements",
  Labeling = "Labeling"
}

export interface ActivityPrescription {
  title: string;
  theory: TherapyTheory;
  durationMinutes: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

export interface LLMResponse {
  message: ChatMessage;
  theoryApplied?: TherapyTheory;
  suggestedActivity?: ActivityPrescription;
}

export interface RAGContext {
  theory: TherapyTheory;
  content: string;
}

export interface MemoryDoc {
  id: string;
  userId: string;
  content: string;
  vector: number[];
}

export enum ThemeName {
  MidnightSanctuary = "MidnightSanctuary",
  DawnAwakening = "DawnAwakening",
  // 6 others...
}

export interface MoodCluster {
  label: string;
  centroid: number[];
  size: number;
}

export enum CrisisLevel {
  None = 0,
  Low = 1,
  Medium = 2,
  High = 3
}

export enum MaslowLevel {
  Physiological = 1,
  Safety = 2,
  LoveAndBelonging = 3,
  Esteem = 4,
  SelfActualization = 5
}
```
