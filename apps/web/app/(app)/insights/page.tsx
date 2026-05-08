'use client';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function InsightsPage() {
  const [driftDismissed, setDriftDismissed] = useState(false);
  const alertLevel = 2; // Mock level

  return (
    <div className="container page-enter" style={{ paddingTop: '3rem', paddingBottom: '4rem', maxWidth: 1000 }}>
      <h2 style={{ marginBottom: '2rem' }}>Psychological Insights</h2>

      {/* Drift Alert */}
      <AnimatePresence>
        {alertLevel >= 2 && !driftDismissed && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95 }}
            className="glass" 
            style={{ 
              padding: '1.5rem', marginBottom: '2rem', 
              background: 'rgba(245, 166, 35, 0.1)', borderColor: 'rgba(245, 166, 35, 0.4)',
              display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start'
            }}
          >
            <div>
              <h4 style={{ color: '#F5A623', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                ⚠️ Mood Drift Detected
              </h4>
              <p style={{ fontSize: '0.9375rem' }}>We've noticed a gradual downward trend in your valence over the last 4 sessions. Consider starting an active prescription.</p>
            </div>
            <button className="btn btn-ghost" onClick={() => setDriftDismissed(true)} style={{ padding: '0.5rem 1rem', minHeight: 'auto' }}>Dismiss</button>
          </motion.div>
        )}
      </AnimatePresence>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        
        {/* Attachment Style */}
        <div className="glass" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
            <h4>Attachment Profile</h4>
            <div className="pill" style={{ background: 'rgba(108, 99, 255, 0.2)', color: '#6C63FF', borderColor: 'rgba(108, 99, 255, 0.4)' }}>Anxious-Preoccupied</div>
          </div>
          <p style={{ marginBottom: '1.5rem' }}>
            You tend to seek high levels of intimacy and approval, and can become overly dependent on others. MIRA adapts to this by:
          </p>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: 8 }}><span>🔹</span> Providing consistent, predictable check-ins</li>
            <li style={{ display: 'flex', gap: 8 }}><span>🔹</span> Explicitly validating emotions before suggesting changes</li>
            <li style={{ display: 'flex', gap: 8 }}><span>🔹</span> Using warm, reassuring language frequently</li>
          </ul>
        </div>

        {/* Cognitive Distortions */}
        <div className="glass" style={{ padding: '2rem' }}>
          <h4 style={{ marginBottom: '1.5rem' }}>Cognitive Distortions (This Week)</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="pill">Catastrophizing</span>
              <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Detected 4 times</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="pill">All-or-Nothing</span>
              <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Detected 2 times</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="pill">Mind Reading</span>
              <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Detected 1 time</span>
            </div>
          </div>
        </div>

      </div>

      <h3 style={{ marginBottom: '1.5rem', marginTop: '3rem' }}>Active Prescriptions</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
        <div className="glass" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <div className="pill" style={{ alignSelf: 'flex-start', marginBottom: '1rem' }}>CBT</div>
          <h4 style={{ marginBottom: '0.5rem' }}>Worry Time Scheduling</h4>
          <p style={{ fontSize: '0.875rem', marginBottom: '1.5rem', flex: 1 }}>Contain worry to a scheduled 15-min window daily. Target: Anxiety.</p>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="btn btn-primary" style={{ flex: 1 }}>Start</button>
            <button className="btn btn-ghost" style={{ flex: 1 }}>Complete</button>
          </div>
        </div>
        <div className="glass" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <div className="pill" style={{ alignSelf: 'flex-start', marginBottom: '1rem', background: 'rgba(45, 203, 117, 0.15)', color: '#2DCB75', borderColor: 'rgba(45,203,117,0.3)' }}>ACT</div>
          <h4 style={{ marginBottom: '0.5rem' }}>Defusion Exercise</h4>
          <p style={{ fontSize: '0.875rem', marginBottom: '1.5rem', flex: 1 }}>Notice thoughts without getting caught in them. Target: Rumination.</p>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="btn btn-primary" style={{ flex: 1 }}>Start</button>
            <button className="btn btn-ghost" style={{ flex: 1 }}>Complete</button>
          </div>
        </div>
      </div>

    </div>
  );
}
