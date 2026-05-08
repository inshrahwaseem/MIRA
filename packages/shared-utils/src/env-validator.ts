import { z } from 'zod';

const clientEnvSchema = z.object({
  NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: z.string().min(1),
  NEXT_PUBLIC_SUPABASE_URL: z.string().url(),
  NEXT_PUBLIC_SUPABASE_ANON_KEY: z.string().min(1),
});

const serverEnvSchema = z.object({
  CLERK_SECRET_KEY: z.string().min(1),
  SUPABASE_SERVICE_ROLE_KEY: z.string().min(1),
  INTERNAL_API_SECRET: z.string().min(16),
  OLLAMA_API_URL: z.string().url(),
  ML_SERVICE_URL: z.string().url(),
  LLM_SERVICE_URL: z.string().url(),
  UPSTASH_REDIS_REST_URL: z.string().url(),
  UPSTASH_REDIS_REST_TOKEN: z.string().min(1),
});

const processEnv = {
  NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY,
  NEXT_PUBLIC_SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL,
  NEXT_PUBLIC_SUPABASE_ANON_KEY: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  CLERK_SECRET_KEY: process.env.CLERK_SECRET_KEY,
  SUPABASE_SERVICE_ROLE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY,
  INTERNAL_API_SECRET: process.env.INTERNAL_API_SECRET,
  OLLAMA_API_URL: process.env.OLLAMA_API_URL,
  ML_SERVICE_URL: process.env.ML_SERVICE_URL,
  LLM_SERVICE_URL: process.env.LLM_SERVICE_URL,
  UPSTASH_REDIS_REST_URL: process.env.UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN: process.env.UPSTASH_REDIS_REST_TOKEN,
};

let clientEnv: z.infer<typeof clientEnvSchema>;
let serverEnv: z.infer<typeof serverEnvSchema>;

try {
  clientEnv = clientEnvSchema.parse(processEnv);
  
  // Only validate server env if we are on the server
  if (typeof window === 'undefined') {
    serverEnv = serverEnvSchema.parse(processEnv);
  }
} catch (error) {
  if (error instanceof z.ZodError) {
    const missingVars = error.issues.map(issue => issue.path.join('.')).join(', ');
    throw new Error(`Missing or invalid required env vars: ${missingVars}`);
  }
  throw error;
}

export const env = {
  ...clientEnv,
  ...(typeof window === 'undefined' ? serverEnv! : {}),
};
