'use client';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { useThemeEngine } from '../../../hooks/useThemeEngine';

export default function SettingsPage() {
  const { setManualTheme, clearManualTheme } = useThemeEngine('neutral');
  const [activeTheme, setActiveTheme] = useState<string | null>(null);

  const themes = [
    { id: 'midnight', name: 'Midnight', color: '#7B6EF6' },
    { id: 'anxious', name: 'Calm', color: '#6C63FF' },
    { id: 'sad', name: 'Warm', color: '#F5A623' },
    { id: 'angry', name: 'Serene', color: '#2DCB75' },
    { id: 'fear', name: 'Breeze', color: '#38B2F0' },
    { id: 'joy', name: 'Radiant', color: '#FFD166' },
    { id: 'inspired', name: 'Inspired', color: '#00BFA5' },
    { id: 'crisis', name: 'Grounding', color: '#BB86FC' },
  ];

  return (
    <div className="container page-enter" style={{ paddingTop: '3rem', paddingBottom: '4rem', maxWidth: 800 }}>
      <h2 style={{ marginBottom: '2rem' }}>Settings</h2>

      {/* Analysis Modes */}
      <section style={{ marginBottom: '3rem' }}>
        <h4 style={{ marginBottom: '1.5rem' }}>Analysis Modes</h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="glass" style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h5 style={{ marginBottom: '0.25rem' }}>Text Analysis</h5>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Core emotion and distortion detection.</p>
            </div>
            <input type="checkbox" defaultChecked style={{ width: 24, height: 24, accentColor: 'var(--accent)' }} />
          </div>
          <div className="glass" style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h5 style={{ marginBottom: '0.25rem' }}>Voice Analysis</h5>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Analyzes tone, pitch, and pacing locally.</p>
            </div>
            <input type="checkbox" style={{ width: 24, height: 24, accentColor: 'var(--accent)' }} />
          </div>
          <div className="glass" style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h5 style={{ marginBottom: '0.25rem' }}>Camera Analysis</h5>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Analyzes micro-expressions locally. Video is NEVER saved.</p>
            </div>
            <input type="checkbox" style={{ width: 24, height: 24, accentColor: 'var(--accent)' }} />
          </div>
        </div>
      </section>

      {/* Theme Gallery */}
      <section style={{ marginBottom: '3rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h4>Theme Gallery</h4>
          <button className="btn btn-ghost" onClick={() => { clearManualTheme(); setActiveTheme(null); }} style={{ minHeight: 32, padding: '4px 12px', fontSize: '0.75rem' }}>
            Auto-Adaptive
          </button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '1rem' }}>
          {themes.map(t => (
            <button 
              key={t.id}
              className="glass"
              onMouseEnter={() => setManualTheme(t.id)}
              onClick={() => setActiveTheme(t.id)}
              style={{ 
                height: 80, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8,
                borderColor: activeTheme === t.id ? 'var(--accent)' : 'var(--border)'
              }}
            >
              <div style={{ width: 24, height: 24, borderRadius: '50%', background: t.color, boxShadow: `0 0 12px ${t.color}` }} />
              <span style={{ fontSize: '0.75rem', fontFamily: 'Syne, sans-serif' }}>{t.name}</span>
            </button>
          ))}
        </div>
      </section>

      {/* Notifications */}
      <section style={{ marginBottom: '3rem' }}>
        <h4 style={{ marginBottom: '1.5rem' }}>Notifications</h4>
        <div className="glass" style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h5 style={{ marginBottom: '0.25rem' }}>Daily Reminder</h5>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Gentle nudge to check in with MIRA.</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <input type="time" defaultValue="20:00" className="input" style={{ width: 'auto', minHeight: 36, padding: '4px 8px' }} />
            <input type="checkbox" defaultChecked style={{ width: 24, height: 24, accentColor: 'var(--accent)' }} />
          </div>
        </div>
      </section>

      {/* Data Management */}
      <section style={{ marginBottom: '3rem' }}>
        <h4 style={{ marginBottom: '1.5rem' }}>Data Management</h4>
        <div className="glass" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>You own your data. MIRA processes sensitive data locally on your device whenever possible.</p>
          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
            <button className="btn btn-ghost" style={{ flex: 1 }}>Export My Data</button>
            <button className="btn btn-ghost" style={{ flex: 1, color: '#ff4d4f', borderColor: 'rgba(255,77,79,0.3)' }}>Delete Account</button>
          </div>
        </div>
      </section>

      {/* About */}
      <section style={{ textAlign: 'center', padding: '2rem 0', opacity: 0.7 }}>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
          MIRA is a supportive AI companion, not a licensed therapist or medical device.<br/>
          If you are in crisis, please call emergency services.
        </p>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem' }}>v1.0.0 Bioluminescent Sanctuary</p>
      </section>

    </div>
  );
}
