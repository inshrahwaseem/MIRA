'use client';
import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import EmotionOrb from '../../../components/emotion-orb/EmotionOrb';
import MessageBubble from '../../../components/chat/MessageBubble';
import TypingIndicator from '../../../components/chat/TypingIndicator';
import VoiceWaveform from '../../../components/chat/VoiceWaveform';
import MoodCalendar from '../../../components/dashboard/MoodCalendar';
import BreathingExercise from '../../../components/breathing/BreathingExercise';

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([
    { id: '1', role: 'assistant' as const, content: 'Assalamu Alaikum. I am MIRA. How are you feeling right now?' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = () => {
    if (!input.trim()) return;
    const newMsg = { id: Date.now().toString(), role: 'user' as const, content: input, emotion: 'Neutral', intensity: 1 as const };
    setMessages(prev => [...prev, newMsg]);
    setInput('');
    setIsTyping(true);
    
    // Mock response
    setTimeout(() => {
      setIsTyping(false);
      setMessages(prev => [...prev, { id: Date.now().toString(), role: 'assistant', content: "I hear you. Let's talk about what's on your mind.", isStreaming: true }]);
    }, 1500);
  };

  // Mock calendar data
  const mockDays = Array.from({ length: 35 }).map((_, i) => ({
    date: `Day ${i+1}`,
    hasData: i < 20,
    valence: i < 20 ? (Math.random() * 2 - 1) : undefined,
    emotion: i < 20 ? ['Joy', 'Fear', 'Sadness', 'Anticipation', 'Neutral'][Math.floor(Math.random() * 5)] : undefined
  }));

  return (
    <div className="three-col">
      {/* LEFT PANEL */}
      <div className="col-left glass" style={{ borderLeft: 'none', borderTop: 'none', borderBottom: 'none', borderRadius: 0, padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <div>
          <h4 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '1.5rem' }}>🔥</span> Streak
          </h4>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.25rem' }}>
            <span style={{ fontFamily: 'DM Serif Display', fontSize: '3rem', lineHeight: 1 }}>12</span>
            <span style={{ color: 'var(--text-muted)' }}>days</span>
          </div>
        </div>
        
        <div>
          <h4 style={{ marginBottom: '1rem' }}>Mood History</h4>
          <MoodCalendar days={mockDays} />
        </div>

        <div style={{ marginTop: 'auto' }}>
          <h4 style={{ marginBottom: '1rem' }}>Modes</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.875rem' }}>Text</span>
              <div className="pill" style={{ background: 'var(--accent)', color: '#fff' }}>Active</div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.875rem' }}>Voice</span>
              <div className="pill">Off</div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.875rem' }}>Camera</span>
              <div className="pill">Off</div>
            </div>
          </div>
        </div>
      </div>

      {/* CENTER PANEL - CHAT */}
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', position: 'relative' }}>
        <div style={{ flex: 1, overflowY: 'auto', padding: '2rem 1.5rem', paddingBottom: '100px' }}>
          <div style={{ maxWidth: 800, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <AnimatePresence initial={false}>
              {messages.map(m => (
                <MessageBubble key={m.id} message={m} />
              ))}
            </AnimatePresence>
            {isTyping && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* INPUT BAR */}
        <div className="glass" style={{ 
          position: 'absolute', bottom: '2rem', left: '2rem', right: '2rem', 
          padding: '0.75rem', borderRadius: '24px', display: 'flex', alignItems: 'center', gap: '0.75rem',
          maxWidth: 800, margin: '0 auto', background: 'var(--bg-surface)'
        }}>
          {isRecording ? (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', padding: '0 1rem' }}>
              <VoiceWaveform isActive={true} />
            </div>
          ) : (
            <input 
              type="text" 
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              placeholder="Type your message..."
              style={{ flex: 1, background: 'transparent', border: 'none', color: '#fff', padding: '0.5rem 1rem', outline: 'none', fontFamily: 'Lato, sans-serif' }}
            />
          )}
          
          <button 
            className="btn btn-ghost" 
            style={{ borderRadius: '50%', width: 44, height: 44, padding: 0 }}
            onMouseDown={() => setIsRecording(true)}
            onMouseUp={() => setIsRecording(false)}
            onMouseLeave={() => setIsRecording(false)}
            aria-label="Hold to record voice"
          >
            🎤
          </button>
          
          <button 
            className="btn btn-primary" 
            style={{ borderRadius: '20px', minHeight: 44, padding: '0 1.5rem' }}
            onClick={handleSend}
          >
            Send
          </button>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div className="col-right glass" style={{ borderRight: 'none', borderTop: 'none', borderBottom: 'none', borderRadius: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '3rem 0', display: 'flex', justifyContent: 'center' }}>
          <EmotionOrb emotion="Neutral" intensity={1} arousal={0.3} size="lg" />
        </div>
        
        <div style={{ padding: '1.5rem', flex: 1 }}>
          <h4 style={{ marginBottom: '1rem' }}>Suggested for you</h4>
          
          <div className="glass" style={{ background: 'var(--bg-secondary)', marginBottom: '1.5rem' }}>
            <BreathingExercise />
          </div>

          <div className="glass" style={{ padding: '1.5rem', background: 'var(--bg-secondary)' }}>
            <div className="pill" style={{ marginBottom: '1rem' }}>CBT Practice</div>
            <h5 style={{ marginBottom: '0.5rem' }}>Thought Record</h5>
            <p style={{ fontSize: '0.875rem', marginBottom: '1.5rem', lineHeight: 1.6 }}>Examine the evidence for and against a negative thought you're having.</p>
            <button className="btn btn-ghost" style={{ width: '100%', fontSize: '0.875rem' }}>Start Exercise</button>
          </div>
        </div>
      </div>
    </div>
  );
}
