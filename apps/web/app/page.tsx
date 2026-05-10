"use client";

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Sparkles, Heart, Shield, ArrowRight } from 'lucide-react';

const containerVariants = {
  initial: { opacity: 0 },
  animate: {
    opacity: 1,
    transition: { staggerChildren: 0.2, delayChildren: 0.3 }
  }
};

const itemVariants = {
  initial: { opacity: 0, y: 30 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] }
  }
};

export default function LandingPage() {
  return (
    <main className="page-enter">
      {/* ─── EmotionOrb Background ─────────────────────────────────── */}
      <motion.div 
        animate={{ 
          scale: [1, 1.1, 1],
          opacity: [0.3, 0.5, 0.3],
          rotate: [0, 180, 360]
        }}
        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full blur-[120px] -z-10"
        style={{ background: 'radial-gradient(circle, var(--accent) 0%, transparent 70%)' }}
      />

      <div className="container min-h-screen flex flex-col items-center justify-center text-center gap-8 py-20">
        <motion.div 
          variants={containerVariants}
          initial="initial"
          animate="animate"
          className="flex flex-col items-center gap-8 max-w-4xl"
        >
          {/* Badge */}
          <motion.div variants={itemVariants}>
            <span className="pill">
              <Sparkles className="w-3 h-3" />
              Pakistan's First Mental Health AI
            </span>
          </motion.div>

          {/* Luxury Heading */}
          <motion.h1 variants={itemVariants} className="glow-text">
            MIRA
          </motion.h1>

          <motion.p variants={itemVariants} className="text-xl max-w-2xl text-[var(--text-secondary)]">
            A multimodal therapeutic companion that understands your emotions through voice, 
            expression, and language. Step into a safe space designed for your growth.
          </motion.p>

          {/* CTA Group */}
          <motion.div variants={itemVariants} className="flex flex-wrap justify-center gap-4">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link href="/sign-up" className="btn btn-primary" aria-label="Start your journey">
                Get Started <ArrowRight className="w-5 h-5 ml-2" />
              </Link>
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link href="/about" className="btn btn-ghost" aria-label="Learn more about MIRA">
                Learn More
              </Link>
            </motion.div>
          </motion.div>

          {/* Feature Grid (Responsive 3-column at 1440px) */}
          <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12 w-full">
            {[
              { icon: <Heart />, title: "Empathetic", desc: "Clinically-grounded AI responses." },
              { icon: <Shield />, title: "Secure", desc: "Your data is strictly confidential." },
              { icon: <Sparkles />, title: "Real-time", desc: "Live emotion detection system." }
            ].map((feature, i) => (
              <div key={i} className="glass p-8 text-left group cursor-default">
                <div className="w-12 h-12 rounded-xl bg-[var(--accent)]/10 flex items-center justify-center text-[var(--accent)] mb-4 group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <h3 className="text-xl mb-2">{feature.title}</h3>
                <p className="text-sm opacity-80">{feature.desc}</p>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </div>

      {/* Screen Reader Only Semantic Content */}
      <section className="sr-only">
        <h2>About MIRA</h2>
        <p>MIRA provides multimodal emotional support using advanced ML and LLM technologies.</p>
      </section>

      {/* Mobile Bottom Nav (Semantic nav) */}
      <nav className="bottom-nav" aria-label="Mobile navigation">
        <div className="bottom-nav__items">
          <div className="bottom-nav__item active">Home</div>
          <div className="bottom-nav__item">Chat</div>
          <div className="bottom-nav__item">Vault</div>
          <div className="bottom-nav__item">Profile</div>
        </div>
      </nav>
    </main>
  );
}
