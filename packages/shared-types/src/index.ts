export enum EmotionName {
  Joy = "Joy", Serenity = "Serenity", Ecstasy = "Ecstasy",
  Trust = "Trust", Acceptance = "Acceptance", Admiration = "Admiration",
  Fear = "Fear", Apprehension = "Apprehension", Terror = "Terror",
  Surprise = "Surprise", Distraction = "Distraction", Amazement = "Amazement",
  Sadness = "Sadness", Pensiveness = "Pensiveness", Grief = "Grief",
  Disgust = "Disgust", Boredom = "Boredom", Loathing = "Loathing",
  Anger = "Anger", Annoyance = "Annoyance", Rage = "Rage",
  Anticipation = "Anticipation", Interest = "Interest", Vigilance = "Vigilance"
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
  DawnAwakening = "DawnAwakening"
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
