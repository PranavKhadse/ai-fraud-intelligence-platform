import React from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  Database,
  TrendingUp,
  BarChart3,
  Layers,
} from 'lucide-react';
import { StatCard } from '../common/StatCard.tsx';
import { Badge } from '../common/Badge.tsx';
import { EmptyState } from '../common/EmptyState.tsx';
import type { MonitoringHealthResponse, MonitoringAlertItem } from '../../types/monitoring.ts';

interface MonitoringOverviewTabProps {
  health: MonitoringHealthResponse;
  onNavigateTab: (tab: 'features' | 'predictions' | 'performance' | 'snapshots') => void;
}

export const MonitoringOverviewTab: React.FC<MonitoringOverviewTabProps> = ({
  health,
  onNavigateTab,
}) => {
  const getStatusBadge = (status: string) => {
    switch (status.toUpperCase()) {
      case 'CRITICAL':
        return <Badge variant="block">CRITICAL</Badge>;
      case 'WARNING':
        return <Badge variant="review">WARNING</Badge>;
      case 'NORMAL':
        return <Badge variant="approve">HEALTHY</Badge>;
      case 'INSUFFICIENT_DATA':
        return <Badge variant="neutral">INSUFFICIENT DATA</Badge>;
      default:
        return <Badge variant="neutral">{status}</Badge>;
    }
  };

  const getAlertIcon = (severity: string) => {
    if (severity.toUpperCase() === 'CRITICAL') {
      return <ShieldAlert size={16} style={{ color: 'var(--status-block)' }} />;
    }
    return <AlertTriangle size={16} style={{ color: 'var(--status-review)' }} />;
  };

  return (
    <div className="monitoring-tab-content">
      {/* 1. Overall System Health Banner */}
      <div className={`monitoring-hero-banner status-glow-${health.overall_status.toLowerCase()}`}>
        <div className="hero-banner-main">
          <div className="hero-status-indicator">
            {health.overall_status === 'CRITICAL' ? (
              <ShieldAlert size={28} style={{ color: 'var(--status-block)' }} />
            ) : health.overall_status === 'WARNING' ? (
              <AlertTriangle size={28} style={{ color: 'var(--status-review)' }} />
            ) : health.overall_status === 'NORMAL' ? (
              <CheckCircle2 size={28} style={{ color: 'var(--status-approve)' }} />
            ) : (
              <Activity size={28} style={{ color: 'var(--text-muted)' }} />
            )}
            <div>
              <div className="hero-headline-row">
                <h2 className="hero-title">Model System Health: {health.overall_status}</h2>
                {getStatusBadge(health.overall_status)}
              </div>
              <p className="hero-subtitle">
                Champion Model v{health.model_version} &bull; Window: {health.window_type} ({new Date(health.window_start).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })} &rarr; {new Date(health.window_end).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })})
              </p>
            </div>
          </div>

          <div className="hero-telemetry-stats">
            <div className="telemetry-pill">
              <span className="lbl">Evaluated Volume:</span>
              <strong className="val font-mono">{health.sample_count.toLocaleString()}</strong>
            </div>
            <div className="telemetry-pill">
              <span className="lbl">Resolved Labeled Cases:</span>
              <strong className="val font-mono">{health.labeled_count.toLocaleString()}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Tri-Engine Component Status Grid */}
      <section aria-label="Monitoring Subsystem Components" style={{ marginTop: '1.25rem' }}>
        <div className="stats-grid">
          <div className="component-clickable-card" onClick={() => onNavigateTab('features')}>
            <StatCard
              title="Feature Data Drift"
              value={health.data_drift_status}
              icon={<Database size={18} />}
              sublabel="55 Canonical Features"
              subvalue="Click to inspect features"
              tone={
                health.data_drift_status === 'CRITICAL'
                  ? 'block'
                  : health.data_drift_status === 'WARNING'
                  ? 'review'
                  : 'approve'
              }
              badge={getStatusBadge(health.data_drift_status)}
            />
          </div>

          <div className="component-clickable-card" onClick={() => onNavigateTab('predictions')}>
            <StatCard
              title="Prediction Score Drift"
              value={health.prediction_drift_status}
              icon={<TrendingUp size={18} />}
              sublabel="Model & Score Shifts"
              subvalue="Click to inspect outputs"
              tone={
                health.prediction_drift_status === 'CRITICAL'
                  ? 'block'
                  : health.prediction_drift_status === 'WARNING'
                  ? 'review'
                  : 'approve'
              }
              badge={getStatusBadge(health.prediction_drift_status)}
            />
          </div>

          <div className="component-clickable-card" onClick={() => onNavigateTab('performance')}>
            <StatCard
              title="Ground-Truth Performance"
              value={health.performance_status}
              icon={<BarChart3 size={18} />}
              sublabel="Threshold 0.78 &amp; Queue Purity"
              subvalue="Click to inspect metrics"
              tone={
                health.performance_status === 'CRITICAL'
                  ? 'block'
                  : health.performance_status === 'WARNING'
                  ? 'review'
                  : 'approve'
              }
              badge={getStatusBadge(health.performance_status)}
            />
          </div>

          <div className="component-clickable-card" onClick={() => onNavigateTab('snapshots')}>
            <StatCard
              title="Active Drift &amp; Perf Alerts"
              value={health.active_alert_count.toString()}
              icon={<Layers size={18} />}
              sublabel="Derived Alert Signals"
              subvalue={health.active_alert_count === 0 ? 'All metrics nominal' : 'Attention required'}
              tone={health.active_alert_count > 0 ? 'review' : 'approve'}
              badge={
                health.active_alert_count > 0 ? (
                  <Badge variant="review">{health.active_alert_count} ACTIVE</Badge>
                ) : (
                  <Badge variant="approve">NOMINAL</Badge>
                )
              }
            />
          </div>
        </div>
      </section>

      {/* 3. Active Derived Alerts Section */}
      <section aria-label="Active Monitoring Alerts" style={{ marginTop: '1.5rem' }}>
        <div className="section-header-row">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={18} className="text-amber" />
            <h3 className="section-title-sm">Active System Drift &amp; Degradation Alerts</h3>
          </div>
          <span className="section-badge-count font-mono">{health.active_alerts.length} Total</span>
        </div>

        {health.active_alerts.length === 0 ? (
          <EmptyState
            title="All Monitoring Signals Nominal"
            description="No feature drift, score divergence, or ground-truth performance degradation alerts detected within the current evaluation window."
            icon={<CheckCircle2 size={28} style={{ color: 'var(--status-approve)' }} />}
          />
        ) : (
          <div className="alerts-table-container">
            <table className="monitoring-table">
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Subsystem</th>
                  <th>Target / Metric</th>
                  <th>Observed Value</th>
                  <th>Threshold / Benchmark</th>
                  <th>Alert Summary</th>
                </tr>
              </thead>
              <tbody>
                {health.active_alerts.map((alert: MonitoringAlertItem, idx: number) => (
                  <tr key={`${alert.component}-${alert.metric_name}-${idx}`}>
                    <td>
                      <span className={`alert-badge severity-${alert.severity.toLowerCase()}`}>
                        {getAlertIcon(alert.severity)}
                        <span>{alert.severity}</span>
                      </span>
                    </td>
                    <td>
                      <span className="component-pill">{alert.component.replace(/_/g, ' ').toUpperCase()}</span>
                    </td>
                    <td>
                      <strong className="font-mono">
                        {alert.feature_name ? `${alert.feature_name}` : alert.metric_name}
                      </strong>
                    </td>
                    <td>
                      <span className="font-mono val-highlight">
                        {typeof alert.observed_value === 'number'
                          ? alert.observed_value < 0.001
                            ? alert.observed_value.toExponential(2)
                            : alert.observed_value.toFixed(4)
                          : alert.observed_value}
                      </span>
                    </td>
                    <td>
                      <span className="font-mono text-muted">
                        {alert.threshold_value !== null && alert.threshold_value !== undefined
                          ? alert.threshold_value.toFixed(4)
                          : 'Configured limit'}
                      </span>
                    </td>
                    <td>
                      <span className="alert-message-text">{alert.message}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
