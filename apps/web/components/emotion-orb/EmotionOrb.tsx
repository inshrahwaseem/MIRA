'use client';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  emotion: string;
  intensity: 1 | 2 | 3;
  arousal: number;
  isActive?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const SIZE_MAP = { sm: 80, md: 140, lg: 220 };

export default function EmotionOrb({ emotion, intensity, arousal, isActive = true, size = 'md' }: Props) {
  const px = SIZE_MAP[size];
  const pulseDuration = Math.max(1.5, 3 - arousal * 1.5);
  const pulseScale = 1 + arousal * 0.07;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
      <motion.div
        animate={isActive ? { scale: [1, pulseScale, 1] } : { scale: 1 }}
        transition={{ repeat: Infinity, duration: pulseDuration, ease: 'easeInOut' }}
        style={{
          width: px, height: px, borderRadius: '50%',
          background: 'radial-gradient(circle at 35% 35%, rgba(255,255,255,0.18), var(--orb-color) 60%, transparent)',
          boxShadow: `0 0 40px var(--orb-color), 0 0 80px rgba(var(--orb-rgb), 0.3), inset 0 0 30px rgba(255,255,255,0.05)`,
          cursor: 'default',
        }}
      />
      <span style={{
        fontFamily: 'Syne, sans-serif', fontSize: 11, fontWeight: 700,
        letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--text-secondary)',
      }}>
        {emotion}
      </span>
      <div style={{ display: 'flex', gap: 6 }}>
        {[1, 2, 3].map(i => (
          <div key={i} style={{
            width: 7, height: 7, borderRadius: '50%',
            background: i <= intensity ? 'var(--orb-color)' : 'var(--bg-surface)',
            border: '1px solid var(--border)',
            boxShadow: i <= intensity ? '0 0 6px var(--orb-color)' : 'none',
            transition: 'all 0.4s ease',
          }} />
        ))}
      </div>
    </div>
  );
}
