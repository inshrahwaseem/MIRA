'use client';

import { useEffect, useCallback } from 'react';

const EMOTION_THEME_MAP: Record<string, string> = {
  Joy: 'joy', Anticipation: 'inspired', Trust: 'midnight',
  Fear: 'fear', Surprise: 'anxious', Sadness: 'sad',
  Disgust: 'angry', Anger: 'angry',
  anxiety: 'anxious', sadness: 'sad', anger: 'angry',
  fear: 'fear', joy: 'joy', neutral: 'midnight',
  inspired: 'inspired', crisis: 'crisis',
};

const STORAGE_KEY = 'mira-theme-override';

export function useThemeEngine(emotion: string) {
  const setTheme = useCallback((theme: string) => {
    document.documentElement.setAttribute('data-theme', theme);
  }, []);

  const setManualTheme = useCallback((theme: string) => {
    localStorage.setItem(STORAGE_KEY, theme);
    setTheme(theme);
  }, [setTheme]);

  const clearManualTheme = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  useEffect(() => {
    // Respect manual user override
    const override = localStorage.getItem(STORAGE_KEY);
    if (override) { setTheme(override); return; }
    const mapped = EMOTION_THEME_MAP[emotion] ?? 'midnight';
    setTheme(mapped);
  }, [emotion, setTheme]);

  return { setManualTheme, clearManualTheme };
}
