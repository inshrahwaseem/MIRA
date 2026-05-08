# Data Models - MIRA

## 1. TypeScript Interfaces (Frontend/Node)

```typescript
/**
 * Core emotion shape following Plutchik's model
 */
export interface EmotionVector {
  emotion: string;       // e.g., "Serenity", "Apprehension"
  intensity: number;     // 0-1
  valence: number;       // -1 to 1
  arousal: number;       // 0 to 1
  dominance: number;     // 0 to 1
}

/**
 * Result of the multimodal fusion
 */
export interface FusedEmotionResult {
  primaryEmotion: string;
  secondaryEmotion: string | null;
  confidence: number;
  conflictDetected: boolean;
  intensityLabel: 'mild' | 'moderate' | 'intense';
  vectors: EmotionVector[];
}

/**
 * Voice features extracted from audio
 */
export interface VoiceAnalysis {
  pitch: { mean: number; std: number; range: number };
  energy: { rmsMean: number; loudness: number };
  temporal: { speechRate: number; pauseDuration: number };
  quality: { jitter: number; shimmer: number; hnr: number };
}

/**
 * Camera landmarks and Action Units
 */
export interface VisualAnalysis {
  microExpressions: { expression: string; confidence: number }[];
  actionUnits: Record<string, number>; // e.g., { "AU1": 0.8, "AU12": 0.4 }
  gaze: { eyeContact: number; blinkRate: number };
  posture: { shoulderSlope: number; headTilt: number; slouchScore: number };
}

/**
 * Session record for persistence
 */
export interface SessionRecord {
  id: string;
  userId: string;
  timestamp: string;
  userInput: string;
  aiResponse: string;
  fusedEmotion: FusedEmotionResult;
  cognitiveDistortions: string[];
  suggestedActivities: string[];
}
```

## 2. Python Pydantic Models (Backend ML)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class EmotionVector(BaseModel):
    emotion: str
    intensity: float = Field(..., ge=0, le=1)
    valence: float = Field(..., ge=-1, le=1)
    arousal: float = Field(..., ge=0, le=1)
    dominance: float = Field(..., ge=0, le=1)

class MultimodalPayload(BaseModel):
    text: str
    audio_features: Optional[Dict[str, float]] = None
    facial_landmarks: Optional[List[List[float]]] = None # MediaPipe 468 pts
    action_units: Optional[Dict[str, float]] = None

class FusionResult(BaseModel):
    primary_emotion: str
    intensity_level: str
    confidence_score: float
    is_conflict: bool
    explanation: str # For SHAP interpretability
    full_vectors: List[EmotionVector]

class PsychologyMetrics(BaseModel):
    distortions_detected: List[str]
    maslow_level: str
    attachment_adaptation: str
    ocean_scores: Dict[str, float]

class MLResponse(BaseModel):
    session_id: str
    fused_emotion: FusionResult
    psychology: PsychologyMetrics
    ai_content: str
    suggested_actions: List[str]
```

## 3. Database Schema (Supabase/PostgreSQL)

### Table: `users`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | uuid | Primary Key (from Clerk) |
| `email` | text | Unique identifier |
| `ocean_profile` | jsonb | Big Five scores |
| `attachment_style` | text | Secure/Anxious/etc. |

### Table: `sessions`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | uuid | Primary Key |
| `user_id` | uuid | Foreign Key -> users.id |
| `created_at` | timestamp | |
| `mood_vector` | vector(64) | For pgvector similarity search |
| `fused_emotion` | jsonb | Full FusionResult |
| `transcript` | text | User + AI content |
| `distortions` | text[] | Array of detected CBT distortions |
