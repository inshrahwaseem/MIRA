import Link from 'next/link';

export default function LandingPage() {
  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: 'radial-gradient(circle at center, #1a1a2e 0%, #0f0f1a 100%)',
      color: '#fff',
      fontFamily: 'Inter, sans-serif',
      textAlign: 'center',
      padding: '2rem'
    }}>
      <h1 style={{ fontSize: '4rem', marginBottom: '1rem', fontFamily: 'DM Serif Display, serif' }}>MIRA</h1>
      <p style={{ fontSize: '1.25rem', color: '#a0a0c0', maxWidth: '600px', marginBottom: '3rem' }}>
        Your Multimodal Mental Health Companion AI. Privacy-first, empathetic, and always here for you.
      </p>
      
      <div style={{ display: 'flex', gap: '1.5rem' }}>
        <Link href="/sign-in" style={{
          padding: '1rem 2.5rem',
          background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
          borderRadius: '30px',
          color: '#fff',
          textDecoration: 'none',
          fontWeight: 'bold',
          transition: 'transform 0.2s'
        }}>
          Get Started
        </Link>
      </div>
      
      <div style={{ marginTop: '4rem', display: 'flex', gap: '2rem', color: '#666' }}>
        <span>Voice Analysis</span>
        <span>•</span>
        <span>Mood Tracking</span>
        <span>•</span>
        <span>Secure & Private</span>
      </div>
    </div>
  );
}
