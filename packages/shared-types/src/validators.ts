import { z } from 'zod';

export const ChatMessageSchema = z.object({
  message: z.string()
    .min(1, "Message cannot be empty")
    .max(1000, "Message cannot exceed 1000 characters")
    .trim(),
  session_id: z.string().uuid("Invalid session ID format")
});

export const EmotionAnalysisSchema = z.object({
  text: z.string()
    .min(1, "Text cannot be empty")
    .max(2000, "Text cannot exceed 2000 characters")
    .trim(),
  audio_base64: z.string().optional()
});

export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type EmotionAnalysis = z.infer<typeof EmotionAnalysisSchema>;
