import React from 'react'

function App() {
  return (
    <div className="min-h-screen bg-secondary flex flex-col items-center justify-center text-text-primary p-4">
      <h1 className="text-4xl md:text-6xl font-display font-bold text-primary mb-4 animate-fade-up">
        Hippo Academy
      </h1>
      <p className="text-lg md:text-xl text-text-secondary text-center max-w-2xl mb-8">
        Platform analitik canggih berbasis AI untuk memprediksi performa video YouTube Anda.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl">
        <div className="bg-surface-elevated border border-[var(--color-border)] p-6 rounded-[var(--radius-md)] shadow-[var(--shadow-card)] hover:border-primary/50 transition-colors">
          <h2 className="text-xl font-semibold text-accent-gold mb-2">Dashboard</h2>
          <p className="text-text-muted text-sm">Visualisasi matrik dan performa channel YouTube Anda secara interaktif.</p>
        </div>
        <div className="bg-surface-elevated border border-[var(--color-border)] p-6 rounded-[var(--radius-md)] shadow-[var(--shadow-glow)] hover:border-primary/50 transition-colors">
          <h2 className="text-xl font-semibold text-accent-teal mb-2">Prediction</h2>
          <p className="text-text-muted text-sm">Gunakan model Machine Learning untuk memprediksi *views* masa depan.</p>
        </div>
        <div className="bg-surface-elevated border border-[var(--color-border)] p-6 rounded-[var(--radius-md)] shadow-[var(--shadow-card)] hover:border-primary/50 transition-colors">
          <h2 className="text-xl font-semibold text-accent-red mb-2">AI Consultant</h2>
          <p className="text-text-muted text-sm">Konsultasikan strategi channel Anda dengan asisten cerdas kami.</p>
        </div>
      </div>
    </div>
  )
}

export default App
