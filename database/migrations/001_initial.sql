-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- TABLE 1: user_profiles
CREATE TABLE user_profiles (
  id UUID PRIMARY KEY,
  attachment_style TEXT CHECK (attachment_style IN ('secure','anxious_preoccupied','dismissive_avoidant','fearful_avoidant')),
  big_five_scores JSONB DEFAULT '{"openness":0.5,"conscientiousness":0.5,"extraversion":0.5,"agreeableness":0.5,"neuroticism":0.5}',
  ocean_confidence FLOAT DEFAULT 0.0,
  maslow_baseline INT CHECK (maslow_baseline BETWEEN 1 AND 5),
  onboarding_complete BOOLEAN DEFAULT FALSE,
  analysis_modes JSONB DEFAULT '{"text":true,"voice":false,"camera":false}',
  preferred_theme TEXT DEFAULT 'midnight',
  streak_count INT DEFAULT 0,
  longest_streak INT DEFAULT 0,
  last_checkin_date DATE,
  total_sessions INT DEFAULT 0,
  weekly_email BOOLEAN DEFAULT TRUE,
  daily_reminder BOOLEAN DEFAULT FALSE,
  reminder_time TIME,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- TABLE 2: sessions
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
  started_at TIMESTAMPTZ DEFAULT NOW(),
  ended_at TIMESTAMPTZ,
  duration_seconds INT,
  modalities_used TEXT[] DEFAULT '{}',
  dominant_emotion TEXT NOT NULL,
  emotion_vector VECTOR(64),
  valence FLOAT CHECK (valence BETWEEN -1.0 AND 1.0),
  arousal FLOAT CHECK (arousal BETWEEN 0.0 AND 1.0),
  dominance FLOAT CHECK (dominance BETWEEN 0.0 AND 1.0),
  intensity_level INT CHECK (intensity_level BETWEEN 1 AND 3),
  cluster_id INT,
  drift_alert_level INT DEFAULT 0 CHECK (drift_alert_level BETWEEN 0 AND 3),
  crisis_level INT DEFAULT 0 CHECK (crisis_level BETWEEN 0 AND 3),
  conflict_detected BOOLEAN DEFAULT FALSE,
  masked_emotion TEXT,
  maslow_level INT CHECK (maslow_level BETWEEN 1 AND 5),
  cognitive_distortions JSONB,
  theory_applied TEXT,
  prescription_given TEXT,
  notes TEXT
);

-- TABLE 3: emotion_scores
CREATE TABLE emotion_scores (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
  emotion_name TEXT NOT NULL,
  plutchik_category TEXT,
  intensity_level INT CHECK (intensity_level BETWEEN 1 AND 3),
  score FLOAT CHECK (score BETWEEN 0.0 AND 1.0),
  source TEXT CHECK (source IN ('text','voice','camera','fused')),
  captured_at TIMESTAMPTZ DEFAULT NOW()
);

-- TABLE 4: prescriptions_given
CREATE TABLE prescriptions_given (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  prescription_name TEXT NOT NULL,
  theory_basis TEXT CHECK (theory_basis IN ('CBT','ACT','DBT','Mindfulness','Behaviourism','Maslow','Attachment')),
  target_emotions TEXT[],
  instructions TEXT NOT NULL,
  difficulty_level INT CHECK (difficulty_level BETWEEN 1 AND 3),
  completed BOOLEAN DEFAULT FALSE,
  user_rating INT CHECK (user_rating BETWEEN 1 AND 5),
  completed_at TIMESTAMPTZ,
  given_at TIMESTAMPTZ DEFAULT NOW()
);

-- TABLE 5: crisis_events (anonymized)
CREATE TABLE crisis_events (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
  session_id UUID REFERENCES sessions(id),
  crisis_level INT CHECK (crisis_level BETWEEN 1 AND 3),
  detection_layers TEXT[],
  triggered_by TEXT,
  response_given TEXT,
  resources_shown BOOLEAN DEFAULT FALSE,
  resolved BOOLEAN,
  followup_sent BOOLEAN DEFAULT FALSE,
  occurred_at TIMESTAMPTZ DEFAULT NOW()
);

-- TABLE 6: conversation_summaries
CREATE TABLE conversation_summaries (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  summary TEXT NOT NULL,
  key_topics TEXT[],
  dominant_emotion TEXT,
  maslow_level INT,
  chroma_doc_id TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- INDEXES
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_started_at ON sessions(started_at DESC);
CREATE INDEX idx_sessions_dominant_emotion ON sessions(dominant_emotion);
CREATE INDEX idx_emotion_scores_session_id ON emotion_scores(session_id);
CREATE INDEX idx_prescriptions_user_id ON prescriptions_given(user_id);
-- HNSW vector index for fast similarity search
CREATE INDEX idx_sessions_emotion_vector ON sessions USING hnsw (emotion_vector vector_cosine_ops);

-- RLS (Row Level Security) — EVERY table
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE emotion_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE prescriptions_given ENABLE ROW LEVEL SECURITY;
ALTER TABLE crisis_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_summaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users_own_data" ON user_profiles FOR ALL USING (id = auth.uid()::uuid);
CREATE POLICY "users_own_sessions" ON sessions FOR ALL USING (user_id = auth.uid()::uuid);
CREATE POLICY "users_own_scores" ON emotion_scores FOR ALL USING (user_id = auth.uid()::uuid);
CREATE POLICY "users_own_prescriptions" ON prescriptions_given FOR ALL USING (user_id = auth.uid()::uuid);
CREATE POLICY "users_own_crises" ON crisis_events FOR ALL USING (user_id = auth.uid()::uuid);
CREATE POLICY "users_own_summaries" ON conversation_summaries FOR ALL USING (user_id = auth.uid()::uuid);

-- TRIGGER: auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql;
CREATE TRIGGER user_profiles_updated_at BEFORE UPDATE ON user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- FUNCTION: calculate streak
CREATE OR REPLACE FUNCTION calculate_streak(p_user_id UUID) RETURNS INT AS $$
DECLARE streak INT := 0; check_date DATE := CURRENT_DATE;
BEGIN
  LOOP
    IF EXISTS (SELECT 1 FROM sessions WHERE user_id = p_user_id AND DATE(started_at) = check_date) THEN
      streak := streak + 1; check_date := check_date - 1;
    ELSE EXIT;
    END IF;
  END LOOP;
  RETURN streak;
END; $$ LANGUAGE plpgsql;

-- VIEW: user mood summary for dashboard
CREATE VIEW user_mood_summary AS
SELECT user_id,
  COUNT(*) FILTER (WHERE started_at >= NOW() - INTERVAL '7 days') AS sessions_this_week,
  AVG(valence) FILTER (WHERE started_at >= NOW() - INTERVAL '7 days') AS avg_valence_week,
  AVG(arousal) FILTER (WHERE started_at >= NOW() - INTERVAL '7 days') AS avg_arousal_week,
  MODE() WITHIN GROUP (ORDER BY dominant_emotion) FILTER (WHERE started_at >= NOW() - INTERVAL '7 days') AS dominant_emotion_week,
  MAX(drift_alert_level) FILTER (WHERE started_at >= NOW() - INTERVAL '7 days') AS max_drift_alert
FROM sessions GROUP BY user_id;

-- FUNCTION: get similar past sessions by cosine similarity
CREATE OR REPLACE FUNCTION get_similar_sessions(p_user_id UUID, p_vector VECTOR(64), p_limit INT DEFAULT 5)
RETURNS TABLE(session_id UUID, dominant_emotion TEXT, similarity FLOAT) AS $$
BEGIN
  RETURN QUERY SELECT id, dominant_emotion, 1 - (emotion_vector <=> p_vector) AS similarity
  FROM sessions WHERE user_id = p_user_id AND emotion_vector IS NOT NULL
  ORDER BY emotion_vector <=> p_vector LIMIT p_limit;
END; $$ LANGUAGE plpgsql;
