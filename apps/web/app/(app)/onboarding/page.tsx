'use client';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
import EmotionOrb from '../../../components/emotion-orb/EmotionOrb';

function TypewriterText({ text, delay = 0, onComplete }: { text: string, delay?: number, onComplete?: () => void }) {
  const [displayText, setDisplayText] = useState('');
  
  useEffect(() => {
    let i = 0;
    const t = setTimeout(() => {
      const interval = setInterval(() => {
        setDisplayText(text.substring(0, i + 1));
        i++;
        if (i >= text.length) {
          clearInterval(interval);
          if (onComplete) onComplete();
        }
      }, 50);
      return () => clearInterval(interval);
    }, delay);
    return () => clearTimeout(t);
  }, [text, delay, onComplete]);

  return <span>{displayText}</span>;
}

export default function OnboardingPage() {
  const [step, setStep] = useState(1);
  const router = useRouter();

  const nextStep = () => setStep(s => Math.min(5, s + 1));

  useEffect(() => {
    if (step === 5) {
      const t = setTimeout(() => router.push('/chat'), 4000);
      return () => clearTimeout(t);
    }
  }, [step, router]);

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Progress Dots */}
      <div style={{ position: 'absolute', top: 40, display: 'flex', gap: 8 }}>
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} style={{
            width: 8, height: 8, borderRadius: '50%',
            background: i <= step ? 'var(--accent)' : 'var(--bg-surface)',
            transition: 'background 0.3s ease'
          }} />
        ))}
      </div>

      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div key="step1" className="page-enter" style={{ textAlign: 'center', maxWidth: 600 }}>
            <h1 style={{ fontSize: 'clamp(2.5rem, 6vw, 3.25rem)', marginBottom: '1rem' }}>Assalamu Alaikum</h1>
            <p style={{ fontSize: '1.25rem', marginBottom: '3rem', opacity: 0.9 }}>
              I am MIRA. I'm here to listen, understand, and support you without judgment.
            </p>
            <button className="btn btn-primary" onClick={nextStep} style={{ padding: '0.875rem 2.5rem', fontSize: '1.125rem' }}>
              Begin
            </button>
            <p style={{ fontSize: '0.875rem', marginTop: '2rem', color: 'var(--text-muted)' }}>
              Disclaimer: MIRA is an AI companion and not a licensed therapist or medical professional. If you are in crisis, please seek immediate professional help.
            </p>
          </motion.div>
        )}

        {step === 2 && (
          <motion.div key="step2" className="page-enter" style={{ maxWidth: 600, width: '100%' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', fontSize: '1.25rem' }}>
              <p><TypewriterText text="How would you like me to address you?" delay={0} /></p>
              <motion.input 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}
                className="input" placeholder="Your name or nickname..." style={{ fontSize: '1.25rem', padding: '1rem' }}
              />
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2 }}>
                <TypewriterText text="What brings you here today?" delay={2000} />
              </motion.p>
              <motion.textarea
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3.5 }}
                className="input" placeholder="Just looking for someone to talk to..." rows={3} style={{ resize: 'none' }}
              />
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 4 }} style={{ textAlign: 'right' }}>
                <button className="btn btn-primary" onClick={nextStep}>Continue</button>
              </motion.div>
            </div>
          </motion.div>
        )}

        {step === 3 && (
          <motion.div key="step3" className="page-enter" style={{ maxWidth: 800, width: '100%', textAlign: 'center' }}>
            <h2 style={{ marginBottom: '1rem' }}>How shall we communicate?</h2>
            <p style={{ marginBottom: '3rem' }}>You can change these at any time in settings.</p>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginBottom: '3rem' }}>
              <div className="glass" style={{ padding: '2rem', textAlign: 'center', borderColor: 'var(--accent)' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>Text</h4>
                <p style={{ fontSize: '0.875rem', marginBottom: '1rem' }}>Chat privately</p>
                <div className="pill" style={{ background: 'var(--accent)', color: '#fff' }}>Enabled</div>
              </div>
              <div className="glass" style={{ padding: '2rem', textAlign: 'center', opacity: 0.7, cursor: 'pointer' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>Voice</h4>
                <p style={{ fontSize: '0.875rem', marginBottom: '1rem' }}>Speak naturally</p>
                <div className="pill">Optional</div>
              </div>
              <div className="glass" style={{ padding: '2rem', textAlign: 'center', opacity: 0.7, cursor: 'pointer' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>Camera</h4>
                <p style={{ fontSize: '0.875rem', marginBottom: '1rem' }}>Expression analysis</p>
                <div className="pill">Optional</div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem' }}>Video is never saved</p>
              </div>
            </div>
            <button className="btn btn-primary" onClick={nextStep}>Next</button>
          </motion.div>
        )}

        {step === 4 && (
          <motion.div key="step4" className="page-enter" style={{ maxWidth: 500, width: '100%' }}>
            <h2 style={{ marginBottom: '2rem', textAlign: 'center' }}>How are you feeling right now?</h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', marginBottom: '3rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontFamily: 'Syne, sans-serif' }}>Mood</span>
                </div>
                <input type="range" min="1" max="10" defaultValue="5" style={{ width: '100%', accentColor: 'var(--accent)' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  <span>Low</span><span>High</span>
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontFamily: 'Syne, sans-serif' }}>Energy</span>
                </div>
                <input type="range" min="1" max="10" defaultValue="5" style={{ width: '100%', accentColor: 'var(--accent)' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  <span>Exhausted</span><span>Energetic</span>
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontFamily: 'Syne, sans-serif' }}>Stress</span>
                </div>
                <input type="range" min="1" max="10" defaultValue="5" style={{ width: '100%', accentColor: 'var(--accent)' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  <span>Calm</span><span>Overwhelmed</span>
                </div>
              </div>
            </div>
            
            <div style={{ textAlign: 'center' }}>
              <button className="btn btn-primary" onClick={nextStep}>Complete</button>
            </div>
          </motion.div>
        )}

        {step === 5 && (
          <motion.div key="step5" className="page-enter" style={{ textAlign: 'center' }}>
            <motion.div 
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 100, damping: 20, delay: 0.2 }}
              style={{ marginBottom: '3rem' }}
            >
              <EmotionOrb emotion="Anticipation" intensity={2} arousal={0.5} size="lg" />
            </motion.div>
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1 }}
            >
              MIRA awaits you
            </motion.h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 2 }}
              style={{ marginTop: '1rem', color: 'var(--text-muted)' }}
            >
              Preparing your sanctuary...
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
