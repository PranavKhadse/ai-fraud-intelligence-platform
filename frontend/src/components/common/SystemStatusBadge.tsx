import React from 'react';
import type { HealthResponse } from '../../types/api.ts';

interface SystemStatusBadgeProps {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
}

export const SystemStatusBadge: React.FC<SystemStatusBadgeProps> = ({
  health,
  loading,
  error,
}) => {
  if (loading) {
    return (
      <div className="system-status-container" title="Checking platform health...">
        <div className="status-indicator-dot loading" />
        <span>Checking Service...</span>
      </div>
    );
  }

  if (error || !health) {
    return (
      <div className="system-status-container" title={`Health Check Error: ${error ?? 'Unreachable'}`}>
        <div className="status-indicator-dot offline" />
        <span>Service Offline</span>
      </div>
    );
  }

  const isOnline = health.status.toLowerCase() === 'healthy' || health.status.toLowerCase() === 'online';

  return (
    <div className="system-status-container" title={`API: ${health.app_name} (v${health.version})`}>
      <div className={`status-indicator-dot ${isOnline ? 'online' : 'offline'}`} />
      <span style={{ fontWeight: 600 }}>{isOnline ? 'Operational' : health.status}</span>
      <span className="system-status-divider">•</span>
      <div className="system-status-meta">
        <span>Model: {health.model_version || 'v1.0.0'}</span>
        <span className="system-status-divider">•</span>
        <span>{health.rules_loaded_count ?? 0} Rules</span>
      </div>
    </div>
  );
};
