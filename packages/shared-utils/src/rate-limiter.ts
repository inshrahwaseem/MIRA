import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";
import { env } from "./env-validator";
import { NextResponse } from "next/server";

// Only initialize if we're on the server and have the env vars
const redis = typeof window === 'undefined' ? new Redis({
  url: env.UPSTASH_REDIS_REST_URL,
  token: env.UPSTASH_REDIS_REST_TOKEN,
}) : null;

// Chat: 20 req/min
export const chatRateLimit = redis ? new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(20, "1 m"),
  analytics: true,
}) : null;

// Analyze: 60 req/min
export const analyzeRateLimit = redis ? new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(60, "1 m"),
  analytics: true,
}) : null;

// Sessions: 100 req/15min
export const sessionsRateLimit = redis ? new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(100, "15 m"),
  analytics: true,
}) : null;

export async function checkRateLimit(
  limitRule: Ratelimit | null, 
  identifier: string
): Promise<NextResponse | null> {
  if (!limitRule) return null; // Fallback if redis not configured (e.g. testing)

  const { success, limit, remaining, reset } = await limitRule.limit(identifier);

  if (!success) {
    return new NextResponse(
      JSON.stringify({ 
        error: "Too many requests", 
        retry_after: Math.ceil((reset - Date.now()) / 1000) 
      }),
      {
        status: 429,
        headers: {
          'Content-Type': 'application/json',
          'X-RateLimit-Limit': limit.toString(),
          'X-RateLimit-Remaining': remaining.toString(),
          'X-RateLimit-Reset': reset.toString()
        }
      }
    );
  }

  // It's successful, we can inject these headers in the final response if we wanted to
  return null;
}
