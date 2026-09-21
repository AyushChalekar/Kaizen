import React from 'react';
import './Bubble.css';

interface BubbleProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'warning' | 'error' | 'success';
}

export function Bubble({ children, className = '', variant = 'default' }: BubbleProps) {
  return (
    <div className={`bubble-container bubble-${variant} ${className}`}>
      <div className="bubble-content">
        {children}
      </div>
    </div>
  );
}
