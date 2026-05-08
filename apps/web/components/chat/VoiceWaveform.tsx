'use client';
import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props { isActive: boolean; }

export default function VoiceWaveform({ isActive }: Props) {
  const barsRef = useRef<HTMLDivElement[]>([]);
  const animRef = useRef<number>();
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataRef = useRef<Uint8Array>(new Uint8Array(6));

  useEffect(() => {
    if (!isActive) { cancelAnimationFrame(animRef.current!); return; }

    let stream: MediaStream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const ctx = new AudioContext();
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 32;
        source.connect(analyser);
        analyserRef.current = analyser;
        dataRef.current = new Uint8Array(analyser.frequencyBinCount);

        const draw = () => {
          analyser.getByteFrequencyData(dataRef.current);
          barsRef.current.forEach((bar, i) => {
            if (!bar) return;
            const val = (dataRef.current[i] ?? 40) / 255;
            const h = Math.max(4, val * 48);
            bar.style.height = `${h}px`;
          });
          animRef.current = requestAnimationFrame(draw);
        };
        draw();
      } catch {
        // Mic not available — animate randomly
        const demo = () => {
          barsRef.current.forEach(bar => {
            if (!bar) return;
            bar.style.height = `${8 + Math.random() * 32}px`;
          });
          animRef.current = requestAnimationFrame(demo);
        };
        demo();
      }
    })();

    return () => {
      cancelAnimationFrame(animRef.current!);
      stream?.getTracks().forEach(t => t.stop());
    };
  }, [isActive]);

  return (
    <AnimatePresence>
      {isActive && (
        <motion.div
          initial={{ opacity: 0, scaleY: 0 }}
          animate={{ opacity: 1, scaleY: 1 }}
          exit={{ opacity: 0, scaleY: 0 }}
          style={{ display: 'flex', alignItems: 'center', gap: 4, height: 56 }}
        >
          {[0, 1, 2, 3, 4, 5].map(i => (
            <div
              key={i}
              ref={el => { if (el) barsRef.current[i] = el; }}
              style={{
                width: 4, height: 16, borderRadius: 99,
                background: `linear-gradient(to top, var(--accent), var(--accent-warm))`,
                transition: 'height 0.08s ease',
                boxShadow: '0 0 8px var(--accent-glow)',
              }}
            />
          ))}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
