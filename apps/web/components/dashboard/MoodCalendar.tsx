'use client';
import { motion } from 'framer-motion';

interface DayData {
  date: string;
  emotion?: string;
  valence?: number;
  hasData: boolean;
}

interface Props { days: DayData[]; }

function getOpacity(valence?: number, hasData?: boolean): number {
  if (!hasData) return 0.12;
  if (valence === undefined) return 0.3;
  if (valence > 0.6) return 1;
  if (valence > 0.2) return 0.8;
  if (valence > -0.2) return 0.6;
  if (valence > -0.6) return 0.4;
  return 0.2;
}

export default function MoodCalendar({ days }: Props) {
  const rows = 5;
  const cols = 7;
  const cells = Array.from({ length: rows * cols }, (_, i) => days[i]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(7, 12px)',
        gap: 3,
      }}>
        {cells.map((day, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: i * 0.01, duration: 0.3 }}
            title={day ? `${day.date}${day.emotion ? ` · ${day.emotion}` : ''}${day.valence !== undefined ? ` (${day.valence.toFixed(2)})` : ''}` : ''}
            style={{
              width: 12, height: 12,
              borderRadius: 3,
              background: `rgba(var(--orb-rgb), ${getOpacity(day?.valence, day?.hasData)})`,
              border: '1px solid rgba(var(--orb-rgb), 0.15)',
              cursor: day?.hasData ? 'pointer' : 'default',
              transition: 'transform 0.15s',
            }}
            whileHover={day?.hasData ? { scale: 1.4 } : {}}
          />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'Syne,sans-serif', letterSpacing: '0.06em' }}>Less</span>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'Syne,sans-serif', letterSpacing: '0.06em' }}>More</span>
      </div>
    </div>
  );
}
