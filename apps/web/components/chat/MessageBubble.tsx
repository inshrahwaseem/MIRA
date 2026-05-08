'use client';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  emotion?: string;
  intensity?: 1 | 2 | 3;
  isStreaming?: boolean;
}

function StreamingText({ text }: { text: string }) {
  const words = text.split(' ');
  return (
    <span>
      {words.map((word, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, x: -4 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.05, duration: 0.15 }}
          style={{ display: 'inline-block', marginRight: '0.25em' }}
        >
          {word}
        </motion.span>
      ))}
    </span>
  );
}

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        gap: 6,
        marginBottom: 4,
      }}
    >
      {!isUser && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
          <div style={{
            width: 36, height: 36, borderRadius: '50%',
            background: 'var(--accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 12, fontWeight: 700, fontFamily: 'Syne, sans-serif',
            color: '#fff', flexShrink: 0,
            boxShadow: '0 0 12px var(--accent-glow)',
          }}>M</div>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'Syne, sans-serif', letterSpacing: '0.06em' }}>
            MIRA
          </span>
        </div>
      )}

      <div style={{
        maxWidth: '78%',
        padding: '12px 16px',
        borderRadius: isUser ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
        background: isUser ? 'var(--accent)' : 'var(--bg-surface)',
        border: isUser ? 'none' : '1px solid var(--border)',
        color: isUser ? '#fff' : 'var(--text-primary)',
        fontFamily: 'Lato, sans-serif',
        fontSize: 15,
        lineHeight: 1.6,
        boxShadow: isUser ? '0 4px 20px var(--accent-glow)' : 'none',
      }}>
        {message.isStreaming
          ? <StreamingText text={message.content} />
          : message.content}
      </div>

      {isUser && message.emotion && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.2 }}
          style={{ display: 'flex', alignItems: 'center', gap: 8, paddingRight: 4 }}
        >
          <span className="pill" style={{ fontSize: 10 }}>{message.emotion}</span>
          {message.intensity && (
            <div style={{ display: 'flex', gap: 3 }}>
              {[1, 2, 3].map(i => (
                <div key={i} style={{
                  width: 5, height: 5, borderRadius: '50%',
                  background: i <= message.intensity! ? 'var(--accent)' : 'var(--border)',
                }} />
              ))}
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}
