'use client';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Technique = '4-7-8' | 'box' | 'natural';

interface Config { inhale: number; hold1: number; exhale: number; hold2: number; label: string; }

const TECHNIQUES: Record<Technique, Config> = {
  'box':     { inhale: 4, hold1: 4, exhale: 4, hold2: 4,  label: 'Box Breathing' },
  '4-7-8':   { inhale: 4, hold1: 7, exhale: 8, hold2: 0,  label: '4-7-8 Calming' },
  'natural': { inhale: 5, hold1: 0, exhale: 6, hold2: 0,  label: 'Natural Calm'  },
};

type Phase = 'inhale' | 'hold1' | 'exhale' | 'hold2' | 'idle';

const PHASE_TEXT: Record<Phase, string> = {
  inhale: 'Breathe in…', hold1: 'Hold…', exhale: 'Breathe out…', hold2: 'Hold…', idle: 'Tap to begin',
};

export default function BreathingExercise() {
  const [technique, setTechnique] = useState<Technique>('box');
  const [phase, setPhase] = useState<Phase>('idle');
  const [isRunning, setIsRunning] = useState(false);
  const cfg = TECHNIQUES[technique];

  useEffect(() => {
    if (!isRunning) { setPhase('idle'); return; }
    const sequence: Array<[Phase, number]> = [
      ['inhale', cfg.inhale],
      ...(cfg.hold1 > 0 ? [['hold1', cfg.hold1] as [Phase, number]] : []),
      ['exhale', cfg.exhale],
      ...(cfg.hold2 > 0 ? [['hold2', cfg.hold2] as [Phase, number]] : []),
    ];

    let idx = 0;
    setPhase(sequence[0][0]);
    const tick = () => {
      idx = (idx + 1) % sequence.length;
      setPhase(sequence[idx][0]);
    };
    const interval = setInterval(tick, sequence[idx][1] * 1000);
    return () => clearInterval(interval);
  }, [isRunning, technique]);

  const circleScale = phase === 'inhale' ? 1.6 : phase === 'exhale' ? 1.0 : phase === 'hold1' ? 1.6 : 1.0;
  const phaseDuration = phase === 'inhale' ? cfg.inhale : phase === 'exhale' ? cfg.exhale : phase === 'hold1' ? cfg.hold1 : cfg.hold2 || 0.3;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 24, padding: '1.5rem' }}>
      {/* Technique selector */}
      <div style={{ display: 'flex', gap: 8 }}>
        {(Object.keys(TECHNIQUES) as Technique[]).map(t => (
          <button
            key={t}
            onClick={() => { setTechnique(t); setIsRunning(false); }}
            className={technique === t ? 'btn btn-primary' : 'btn btn-ghost'}
            style={{ padding: '6px 14px', fontSize: 12, minHeight: 36 }}
            aria-label={`Select ${TECHNIQUES[t].label}`}
          >
            {t === 'box' ? 'Box' : t === '4-7-8' ? '4-7-8' : 'Natural'}
          </button>
        ))}
      </div>

      {/* Breathing circle */}
      <motion.div
        animate={{ scale: phase === 'idle' ? 1 : circleScale }}
        transition={{ duration: phaseDuration, ease: 'easeInOut' }}
        onClick={() => setIsRunning(r => !r)}
        style={{
          width: 160, height: 160, borderRadius: '50%', cursor: 'pointer',
          background: 'radial-gradient(circle at 35% 35%, rgba(255,255,255,0.15), var(--orb-color) 70%)',
          boxShadow: '0 0 40px var(--orb-color), 0 0 80px rgba(var(--orb-rgb), 0.25)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <motion.span
          key={phase}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{ fontFamily: 'Syne, sans-serif', fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', color: '#fff', textAlign: 'center', padding: '0 16px' }}
        >
          {PHASE_TEXT[phase]}
        </motion.span>
      </motion.div>

      <p style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'Syne,sans-serif', letterSpacing: '0.06em' }}>
        {isRunning ? cfg.label : 'Tap to start'}
      </p>
    </div>
  );
}
