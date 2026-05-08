'use client';
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import MoodCalendar from '../../../components/dashboard/MoodCalendar';

function CountUp({ end, suffix = '' }: { end: number, suffix?: string }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = 0;
    const duration = 1500;
    const startTime = performance.now();
    const update = (t: number) => {
      const elapsed = t - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutExpo
      const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      setVal(Math.floor(ease * end));
      if (progress < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }, [end]);
  return <span>{val}{suffix}</span>;
}

export default function DashboardPage() {
  const mockDays = Array.from({ length: 30 }).map((_, i) => ({
    date: `Day ${i+1}`,
    hasData: i > 5,
    valence: i > 5 ? (Math.random() * 2 - 1) : undefined,
    emotion: i > 5 ? 'Anticipation' : undefined
  }));

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } }
  };

  return (
    <motion.div 
      className="container page-enter" 
      style={{ paddingTop: '3rem', paddingBottom: '4rem' }}
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      <motion.h2 variants={itemVariants} style={{ marginBottom: '2rem' }}>Your Journey</motion.h2>
      
      {/* ROW 1: STATS */}
      <motion.div variants={itemVariants} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.5rem', marginBottom: '3rem' }}>
        <div className="glass stat-card">
          <span className="stat-card__label">Total Sessions</span>
          <span className="stat-card__value"><CountUp end={42} /></span>
        </div>
        <div className="glass stat-card">
          <span className="stat-card__label">Avg Mood Valence</span>
          <span className="stat-card__value" style={{ color: 'var(--accent)' }}>+<CountUp end={68} suffix="%" /></span>
        </div>
        <div className="glass stat-card">
          <span className="stat-card__label">Current Streak</span>
          <span className="stat-card__value"><CountUp end={12} suffix=" days" /> 🔥</span>
        </div>
      </motion.div>

      {/* ROW 2: CALENDAR */}
      <motion.div variants={itemVariants} className="glass" style={{ padding: '2rem', marginBottom: '3rem' }}>
        <h4 style={{ marginBottom: '1.5rem' }}>Mood Calendar (30 Days)</h4>
        <div style={{ overflowX: 'auto' }}>
          <MoodCalendar days={mockDays} />
        </div>
      </motion.div>

      {/* ROW 3: CHARTS */}
      <motion.div variants={itemVariants} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '3rem' }}>
        <div className="glass" style={{ padding: '2rem', minHeight: 300, display: 'flex', flexDirection: 'column' }}>
          <h4 style={{ marginBottom: '1rem' }}>Emotion Trend</h4>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 12, position: 'relative', overflow: 'hidden' }}>
             <svg width="100%" height="100%" viewBox="0 0 400 200" preserveAspectRatio="none">
               <motion.path 
                 d="M0,150 Q50,50 100,100 T200,80 T300,120 T400,40"
                 fill="none" 
                 stroke="var(--accent)" 
                 strokeWidth="4"
                 initial={{ pathLength: 0 }}
                 animate={{ pathLength: 1 }}
                 transition={{ duration: 2, ease: "easeOut" }}
                 style={{ filter: 'drop-shadow(0 0 8px var(--accent))' }}
               />
               <motion.path 
                 d="M0,150 Q50,50 100,100 T200,80 T300,120 T400,40 L400,200 L0,200 Z"
                 fill="var(--accent-glow)" 
                 initial={{ opacity: 0 }}
                 animate={{ opacity: 1 }}
                 transition={{ duration: 2, ease: "easeOut" }}
               />
             </svg>
          </div>
        </div>

        <div className="glass" style={{ padding: '2rem', minHeight: 300, display: 'flex', flexDirection: 'column' }}>
          <h4 style={{ marginBottom: '1rem' }}>Emotion Radar</h4>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 12, position: 'relative' }}>
            <svg width="200" height="200" viewBox="0 0 200 200">
               {/* Background Grid */}
               <polygon points="100,10 185,55 185,145 100,190 15,145 15,55" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
               <polygon points="100,35 160,70 160,130 100,165 40,130 40,70" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
               <polygon points="100,60 135,85 135,115 100,140 65,115 65,85" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
               <motion.polygon 
                 points="100,20 180,70 150,160 50,160 20,70"
                 fill="var(--accent-glow)" 
                 stroke="var(--accent)" 
                 strokeWidth="2"
                 initial={{ scale: 0, opacity: 0 }}
                 animate={{ scale: 1, opacity: 1 }}
                 transition={{ duration: 1, delay: 0.5, type: 'spring' }}
               />
            </svg>
          </div>
        </div>
      </motion.div>

      {/* ROW 4: CLUSTERS & PATTERNS */}
      <motion.div variants={itemVariants} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem' }}>
        <div className="glass" style={{ padding: '2rem', minHeight: 300 }}>
          <h4 style={{ marginBottom: '1rem' }}>Cluster Scatter (t-SNE)</h4>
          <div style={{ height: 200, background: 'rgba(0,0,0,0.2)', borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
            {[...Array(25)].map((_, i) => (
              <motion.div key={`c1-${i}`} 
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1 + i * 0.05 }}
                style={{ 
                  position: 'absolute', 
                  left: `${20 + Math.random() * 20}%`, 
                  top: `${20 + Math.random() * 20}%`, 
                  width: 8, height: 8, borderRadius: '50%', 
                  background: 'var(--accent)',
                  boxShadow: '0 0 10px var(--accent-glow)'
                }} 
              />
            ))}
            {[...Array(15)].map((_, i) => (
              <motion.div key={`c2-${i}`} 
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 2 + i * 0.05 }}
                style={{ 
                  position: 'absolute', 
                  left: `${60 + Math.random() * 20}%`, 
                  top: `${50 + Math.random() * 30}%`, 
                  width: 8, height: 8, borderRadius: '50%', 
                  background: 'var(--accent-warm)',
                  boxShadow: '0 0 10px rgba(var(--orb-rgb), 0.5)'
                }} 
              />
            ))}
          </div>
        </div>

        <div className="glass" style={{ padding: '2rem', minHeight: 300 }}>
          <h4 style={{ marginBottom: '1.5rem' }}>Trigger Patterns</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
            <div className="pill" style={{ padding: '0.5rem 1rem' }}>Work Deadlines (85%)</div>
            <div className="pill" style={{ padding: '0.5rem 1rem', background: 'rgba(245,166,35,0.15)', color: '#F5A623', borderColor: 'rgba(245,166,35,0.3)' }}>Poor Sleep (72%)</div>
            <div className="pill" style={{ padding: '0.5rem 1rem', background: 'rgba(45,203,117,0.15)', color: '#2DCB75', borderColor: 'rgba(45,203,117,0.3)' }}>Nature Walks (Positive - 90%)</div>
            <div className="pill" style={{ padding: '0.5rem 1rem' }}>Social Media (60%)</div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
