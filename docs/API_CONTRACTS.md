# API Contracts - MIRA

## 1. Authentication
All requests must include a `Authorization: Bearer <clerk_token>` header, verified by the Express.js gateway.

## 2. Endpoints

### POST `/api/v1/session/analyze`
The main entry point for real-time analysis. This is a streaming endpoint (SSE).

**Request Body:**
```json
{
  "sessionId": "uuid",
  "text": "I've been feeling a bit overwhelmed lately, but I'm doing okay.",
  "audioFeatures": {
    "mfcc": [...],
    "pitch": 210.5,
    "energy": 0.045
  },
  "visualFeatures": {
    "actionUnits": { "AU1": 0.7, "AU4": 0.8 },
    "landmarks": [...]
  }
}
```

**Response (SSE Stream):**
*   **Event: `emotion`**
    ```json
    { "primary": "Apprehension", "intensity": "moderate", "conflict": true }
    ```
*   **Event: `psychology`**
    ```json
    { "distortions": ["Catastrophizing"], "theory": "CBT" }
    ```
*   **Event: `content`**
    ```json
    { "chunk": "It sounds like you're carrying a lot right now..." }
    ```
*   **Event: `done`**
    ```json
    { "summaryId": "uuid" }
    ```

---

### GET `/api/v1/history/mood-map`
Retrieves data for the 3D Emotion Universe.

**Query Params:**
*   `period`: `30d`, `90d`, `all` (default `30d`)

**Response:**
```json
[
  {
    "timestamp": "2024-05-01T10:00:00Z",
    "x": 0.45,
    "y": -0.2,
    "z": 0.8,
    "emotion": "Serenity"
  }
]
```

---

### GET `/api/v1/insights/profile`
Retrieves the Big Five and Attachment Style analysis.

**Response:**
```json
{
  "attachmentStyle": "Anxious-Preoccupied",
  "ocean": {
    "openness": 0.8,
    "conscientiousness": 0.6,
    "extraversion": 0.4,
    "agreeableness": 0.9,
    "neuroticism": 0.7
  },
  "topDistortions": ["Black-and-White Thinking", "Labeling"]
}
```

---

### POST `/api/v1/tools/reframing`
Interactive CBT thought record assistant.

**Request:**
```json
{
  "situation": "My boss didn't reply to my email.",
  "automaticThought": "He's going to fire me.",
  "emotion": "Panic"
}
```

**Response:**
```json
{
  "distortionDetected": "Fortune Telling",
  "socraticQuestion": "What evidence do you have that he's going to fire you, other than the delayed reply?",
  "reframingSuggestion": "He might be in back-to-back meetings or focusing on a deadline."
}
```

---

## 3. Error Codes

| Code | Status | Description |
| :--- | :--- | :--- |
| `AUTH_EXPIRED` | 401 | Clerk token has expired. |
| `ML_TIMEOUT` | 504 | Python ML service did not respond in time. |
| `RATE_LIMIT` | 429 | Upstash Redis threshold reached. |
| `CRISIS_TRIGGER` | 200* | Special handling: Returns crisis payload instead of normal AI content. |
| `SENSORS_OFFLINE` | 400 | Required multimodal data missing for selected mode. |
