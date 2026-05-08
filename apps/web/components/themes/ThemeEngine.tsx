'use client';

import { ReactNode } from 'react';
import { useThemeEngine } from '../../hooks/useThemeEngine';

export default function ThemeEngine({ children, initialEmotion = 'neutral' }: { children: ReactNode, initialEmotion?: string }) {
  // Initialize theme tracking
  useThemeEngine(initialEmotion);
  
  return <>{children}</>;
}
