import React from 'react';
import type { RiskTier } from '../../types/api.ts';

interface RiskScoreMeterProps {
  score: number;
  tier?: RiskTier;
  compact?: boolean;
}

export const RiskScoreMeter: React.FC<RiskScoreMeterProps> = ({
  score,
  tier,
  compact = false,
}) => {
  const normalizedScore = Math.max(0, Math.min(100, Math.round(score)));

  const getTierColor = (val: number, specifiedTier?: RiskTier): string => {
    if (specifiedTier) {
      switch (specifiedTier) {
        case 'LOW':
          return 'var(--status-approve)';
        case 'MEDIUM':
          return 'var(--status-review)';
        case 'HIGH':
          return '#F97316';
        case 'CRITICAL':
          return 'var(--status-block)';
      }
    }
    if (val < 35) return 'var(--status-approve)';
    if (val < 60) return 'var(--status-review)';
    if (val < 78) return '#F97316';
    return 'var(--status-block)';
  };

  const color = getTierColor(normalizedScore, tier);

  return (
    <div
      className={`risk-score-meter ${compact ? 'compact' : ''}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        minWidth: compact ? '80px' : '110px',
      }}
    >
      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontWeight: 700,
          fontSize: '0.8125rem',
          color: color,
          width: '24px',
          textAlign: 'right',
        }}
      >
        {normalizedScore}
      </span>
      <div
        style={{
          flex: 1,
          height: '6px',
          backgroundColor: 'var(--bg-card-hover)',
          borderRadius: 'var(--radius-full)',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <div
          style={{
            width: `${normalizedScore}%`,
            height: '100%',
            backgroundColor: color,
            borderRadius: 'var(--radius-full)',
            transition: 'width 0.3s ease-in-out',
          }}
        />
      </div>
    </div>
  );
};
