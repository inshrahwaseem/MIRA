# Product Requirements Document (PRD) - MIRA

**Project Name:** MIRA — Multimodal Mental Health Companion AI  
**Tagline:** Your personal AI psychiatrist — listens, sees, and understands.  
**Version:** 1.0.0  
**Status:** Planning

---

## 1. Problem Statement
Modern mental health support is often inaccessible, expensive, or delayed. Text-only AI companions lack the emotional intelligence to recognize non-verbal cues (voice tone, facial expressions, posture), leading to flat interactions that fail to identify genuine distress or cognitive distortions. There is a critical need for a privacy-first, multimodal companion that bridges the gap between digital convenience and human-like emotional perception.

## 2. User Personas

### Persona 1: "Anxious Alex" (The Overwhelmed Professional)
*   **Profile:** 28-year-old software engineer working remotely.
*   **Pain Points:** High stress, frequent "catastrophizing," social isolation.
*   **Goal:** Wants a safe space to vent and receive evidence-based reframing (CBT) without the friction of scheduling a therapist.
*   **Usage:** Uses MIRA for 5-minute mid-day "check-ins" and "thought records" when feeling overwhelmed.

### Persona 2: "Melancholic Maya" (The Emotional Navigator)
*   **Profile:** 45-year-old navigating a major life transition (empty nest).
*   **Pain Points:** Difficulty labeling complex emotions, feeling unheard, "emotional blunting."
*   **Goal:** Needs help identifying why she feels "numb" and wants to explore attachment patterns.
*   **Usage:** Deep evening sessions using camera and voice to track mood drift and attachment-style insights.

### Persona 3: "Steady Sam" (The Growth Mindset Seeker)
*   **Profile:** 20-year-old student interested in self-optimization and ACT (Acceptance and Commitment Therapy).
*   **Pain Points:** Procrastination, lack of clarity on values, occasional "black and white" thinking.
*   **Goal:** Wants to build a "3D Emotion Universe" to visualize long-term growth and practice mindfulness.
*   **Usage:** Daily gamified check-ins and breathing exercises tied to physiological stress detection.

## 3. MoSCoW Features

### Must Have (P0)
*   **Multimodal Emotion Fusion:** Bayesian fusion of Text (DistilRoBERTa), Voice (Librosa/Whisper), and Camera (MediaPipe/DeepFace).
*   **Privacy-First Local LLM:** Ollama (Llama 3) for all sensitive processing.
*   **Crisis Protocol:** Layered keyword and ML-based crisis detection with warm response.
*   **CBT Psychology Engine:** Detection of 8 cognitive distortions and Socratic questioning.
*   **Emotion Orb:** Real-time visual feedback of emotional arousal and valence.

### Should Have (P1)
*   **RAG Knowledge Base:** Integration of Beck (CBT), Harris (ACT), and Bowlby (Attachment).
*   **Voice Biomarkers:** Detection of clinical markers like speech rate and pause duration.
*   **Action Unit Mapping:** Ekman micro-expression detection via Action Units (AU1, 4, 6, etc.).
*   **3D Emotion Universe:** Visualization of session history using Plotly/Three.js.
*   **Long-term Memory:** ChromaDB-backed RAG for personalized history and triggers.

### Could Have (P2)
*   **Big Five Personality Profiler:** Automated OCEAN scoring based on conversation logs.
*   **Trigger Pattern Mining:** Apriori algorithm to find correlations (e.g., "work + alone = anxiety").
*   **Monthly Insight Reports:** Auto-generated summaries of emotional trends and progress.
*   **Gamified Streaks:** Check-in incentives to maintain usage consistency.

### Won't Have (P3)
*   **Clinical Diagnosis:** MIRA does not issue DSM-5 diagnoses.
*   **Medication Advice:** Zero recommendations for pharmaceutical interventions.
*   **Live Human Handoff:** Fully automated; no real-time connection to human doctors.

## 4. User Stories
*   *As a user,* I want to talk to MIRA so that it can detect my anxiety through my voice even if I say "I'm fine."
*   *As a user,* I want my data processed locally so that I don't have to worry about my private thoughts leaking.
*   *As a user,* I want to visualize my mood history in 3D so that I can see patterns I wasn't aware of.
*   *As a user,* I want MIRA to catch when I'm "catastrophizing" and help me reframe my thoughts using CBT.

## 5. Ethical Constraints & Safety
*   **Non-Clinical Disclaimer:** Must be prominently displayed on every session start.
*   **Privacy-First:** Raw camera frames and audio clips are NEVER saved; only extracted features (Action Units, MFCCs) are used.
*   **Crisis Protocol:** If high risk is detected, MIRA must provide localized helpline numbers and a non-alarming de-escalation prompt.
*   **No Manipulation:** AI must not use "dark patterns" or addictive loops to keep users engaged.

## 6. Success Metrics
*   **Accuracy:** >85% agreement between fused emotion and user self-report.
*   **Engagement:** Average 3+ check-ins per week per user.
*   **Value:** 70% of users report "highly helpful" reframing for cognitive distortions.
*   **Performance:** Local LLM response latency < 2 seconds.
*   **Safety:** 100% detection rate for high-risk keyword triggers.
