'use client';
import { motion } from 'framer-motion';

export default function TypingIndicator() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '10px 16px', width: 'fit-content' }}>
      {[0, 1, 2].map(i => (
        <motion.div
          key={i}
          animate={{ y: [-4, 0, -4] }}
          transition={{ repeat: Infinity, duration: 1, delay: i * 0.2, ease: 'easeInOut' }}
          style={{
            width: 8, height: 8, borderRadius: '50%',
            background: 'var(--accent)',
            opacity: 0.7,
          }}
        />
      ))}
    </div>
  );
}
