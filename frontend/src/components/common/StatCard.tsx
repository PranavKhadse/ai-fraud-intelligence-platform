import React from 'react';

export type StatCardTone = 'approve' | 'review' | 'block' | 'primary' | 'info';

interface StatCardProps {
  title: string;
  value: string | number;
  icon?: React.ReactNode;
  subvalue?: string | number;
  sublabel?: string;
  tone?: StatCardTone;
  badge?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  icon,
  subvalue,
  sublabel,
  tone,
  badge,
}) => {
  return (
    <div className="stat-card" data-tone={tone}>
      <div>
        <div className="stat-card-header">
          <span className="stat-card-title">{title}</span>
          {icon && <div className="stat-card-icon">{icon}</div>}
        </div>
        <div className="stat-card-value">{value}</div>
      </div>
      {(subvalue !== undefined || badge) && (
        <div className="stat-card-footer">
          {sublabel && <span>{sublabel}</span>}
          {subvalue !== undefined && <span className="stat-card-subvalue">{subvalue}</span>}
          {badge && <div>{badge}</div>}
        </div>
      )}
    </div>
  );
};
