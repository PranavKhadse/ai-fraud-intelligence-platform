import React, { useEffect, useState, useCallback } from 'react';
import {
  BarChart2,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  ShieldCheck,
  Percent,
  Inbox,
  Activity,
} from 'lucide-react';
import { fetchModelPerformance } from '../../api/monitoringApi.ts';
import { ConfusionMatrixGrid } from './ConfusionMatrixGrid.tsx';
import { StatCard } from '../common/StatCard.tsx';
import { Badge } from '../common/Badge.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { EmptyState } from '../common/EmptyState.tsx';
import type {
  MetricDegradationItemResponse,
  ModelPerformanceResponse,
  MonitoringTimeWindow,
} from '../../types/monitoring.ts';

interface ModelPerformanceTabProps {
  window: MonitoringTimeWindow;
  startTime?: string;
  endTime?: string;
  modelVersion?: string;
}

export const ModelPerformanceTab: React.FC<ModelPerformanceTabProps> = ({
  window,
  startTime,
  endTime,
  modelVersion = '1.0.0',
}) => {
  const [data, setData] = useState<ModelPerformanceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedThreshold, setSelectedThreshold] = useState<number>(0.78);

  const loadPerformance = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchModelPerformance({
        window,
        start_time: startTime,
        end_time: endTime,
        operating_threshold: selectedThreshold,
        model_version: modelVersion,
      });
      setData(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch model performance metrics');
    } finally {
      setLoading(false);
    }
  }, [window, startTime, endTime, selectedThreshold, modelVersion]);

  useEffect(() => {
    loadPerformance();
  }, [loadPerformance]);

  const getConfidenceBadge = (conf: string) => {
    switch (conf) {
      case 'NORMAL_CONFIDENCE':
        return <Badge variant="approve">HIGH CONFIDENCE (N &ge; 100)</Badge>;
      case 'LOW_SAMPLE':
        return <Badge variant="review">LOW SAMPLE (20 &le; N &lt; 100)</Badge>;
      case 'INSUFFICIENT_DATA':
        return <Badge variant="neutral">INSUFFICIENT DATA (N &lt; 20)</Badge>;
      default:
        return <Badge variant="neutral">{conf}</Badge>;
    }
  };

  const getSeverityBadge = (status: string) => {
    switch (status.toUpperCase()) {
      case 'CRITICAL':
        return <Badge variant="block">CRITICAL</Badge>;
      case 'WARNING':
        return <Badge variant="review">WARNING</Badge>;
      case 'NORMAL':
        return <Badge variant="approve">NOMINAL</Badge>;
      default:
        return <Badge variant="neutral">{status}</Badge>;
    }
  };

  const activeMetrics =
    data && selectedThreshold === 0.50 ? data.comparison_metrics : data?.primary_metrics;

  return (
    <div className="monitoring-tab-content">
      {error && (
        <ErrorBanner
          title="Performance Telemetry Error"
          message={error}
          onRetry={loadPerformance}
        />
      )}

      {loading ? (
        <LoadingSpinner message="Evaluating ground-truth case dispositions, operating threshold metrics, and degradation deltas..." />
      ) : data ? (
        <>
          {/* 1. Ground-Truth Sample Size & Confidence Header */}
          <div className="performance-confidence-banner">
            <div className="confidence-left">
              <div className="confidence-title-row">
                <h3 className="confidence-title">Ground-Truth Performance Monitoring</h3>
                {getConfidenceBadge(data.confidence)}
              </div>
              <span className="confidence-meta">
                Joined via Resolved Cases &bull; Model v{data.model_version} &bull; Window: {data.window_type}
              </span>
            </div>

            <div className="sample-breakdown-tags">
              <div className="tag-item">
                <span className="lbl">Labeled Cases:</span>
                <strong className="val font-mono">{data.labeled_sample_count.toLocaleString()}</strong>
              </div>
              <div className="tag-item">
                <span className="lbl">Confirmed Fraud:</span>
                <strong className="val font-mono text-red">{data.fraud_cases_count.toLocaleString()}</strong>
              </div>
              <div className="tag-item">
                <span className="lbl">Legitimate:</span>
                <strong className="val font-mono text-green">{data.legitimate_cases_count.toLocaleString()}</strong>
              </div>
            </div>
          </div>

          {data.labeled_sample_count < 20 ? (
            <div style={{ marginTop: '1.25rem' }}>
              <EmptyState
                title="Insufficient Labeled Cases for Statistical Performance"
                description={`Only ${data.labeled_sample_count} resolved ground-truth cases recorded in this window. A minimum of 20 resolved cases is required to compute degradation signals, and 100+ for normal confidence alerts.`}
                icon={<Activity size={28} />}
              />
            </div>
          ) : (
            <>
              {/* 2. Operating Threshold Selector & Model Classification KPIs */}
              <div className="threshold-toggle-bar" style={{ marginTop: '1.25rem' }}>
                <span className="toggle-label">Operating Decision Point:</span>
                <div className="tab-pill-group">
                  <button
                    type="button"
                    className={`tab-pill ${selectedThreshold === 0.78 ? 'active' : ''}`}
                    onClick={() => setSelectedThreshold(0.78)}
                  >
                    Default Threshold (0.78)
                  </button>
                  <button
                    type="button"
                    className={`tab-pill ${selectedThreshold === 0.50 ? 'active' : ''}`}
                    onClick={() => setSelectedThreshold(0.50)}
                  >
                    Comparison Threshold (0.50)
                  </button>
                </div>
              </div>

              {activeMetrics && (
                <div className="stats-grid" style={{ marginTop: '1rem' }}>
                  <StatCard
                    title="Precision (PPV)"
                    value={`${(activeMetrics.precision * 100).toFixed(1)}%`}
                    icon={<Percent size={18} />}
                    sublabel={`Threshold: ${activeMetrics.threshold.toFixed(2)}`}
                    subvalue="Fraud prediction purity"
                    tone="approve"
                    badge={<Badge variant="approve">PRECISION</Badge>}
                  />

                  <StatCard
                    title="Recall (Sensitivity)"
                    value={`${(activeMetrics.recall * 100).toFixed(1)}%`}
                    icon={<ShieldCheck size={18} />}
                    sublabel="Fraud Capture Rate"
                    subvalue={`${activeMetrics.confusion_matrix.tp} of ${data.fraud_cases_count} caught`}
                    tone="primary"
                    badge={<Badge variant="info">RECALL</Badge>}
                  />

                  <StatCard
                    title="F1-Score (Harmonic Mean)"
                    value={activeMetrics.f1.toFixed(3)}
                    icon={<Activity size={18} />}
                    sublabel="Balanced Metric"
                    subvalue={`Accuracy: ${(activeMetrics.accuracy * 100).toFixed(1)}%`}
                    tone="info"
                    badge={<Badge variant="info">F1</Badge>}
                  />

                  <StatCard
                    title="False Positive Rate (FPR)"
                    value={`${(activeMetrics.fpr * 100).toFixed(2)}%`}
                    icon={<AlertTriangle size={18} />}
                    sublabel="Target SLA: &le; 1.0%"
                    subvalue={`${activeMetrics.confusion_matrix.fp} false alerts`}
                    tone={activeMetrics.fpr > 0.02 ? 'review' : 'approve'}
                    badge={
                      <Badge variant={activeMetrics.fpr > 0.02 ? 'review' : 'approve'}>
                        {activeMetrics.fpr > 0.02 ? 'ELEVATED' : 'SLA PASS'}
                      </Badge>
                    }
                  />
                </div>
              )}

              {/* 3. Confusion Matrix & Operational Purity */}
              <div className="performance-details-grid" style={{ marginTop: '1.5rem' }}>
                {/* Confusion Matrix */}
                {activeMetrics && (
                  <div className="analytics-card">
                    <ConfusionMatrixGrid
                      matrix={activeMetrics.confusion_matrix}
                      threshold={activeMetrics.threshold}
                    />
                  </div>
                )}

                {/* Operational Decision Metrics & Review Queue Purity */}
                <div className="analytics-card operational-purity-card">
                  <div className="section-sub-header">
                    <div>
                      <h4 className="sub-title">Operational Decision &amp; Queue Purity</h4>
                      <span className="sub-meta">Hybrid Rule + ML Decision Interventions</span>
                    </div>
                  </div>

                  <div className="operational-kpi-list" style={{ marginTop: '1rem' }}>
                    <div className="op-kpi-item">
                      <div className="op-kpi-label">
                        <Inbox size={16} className="text-amber" />
                        <span>Review Queue Purity:</span>
                      </div>
                      <strong className="op-kpi-val font-mono">
                        {(data.operational_metrics.review_queue_purity * 100).toFixed(1)}%
                      </strong>
                      <span className="op-kpi-sub font-mono">
                        ({data.operational_metrics.fraud_in_review_count} /{' '}
                        {data.operational_metrics.total_reviews_count} escalated cases were confirmed fraud)
                      </span>
                    </div>

                    <div className="op-kpi-item">
                      <div className="op-kpi-label">
                        <ShieldAlert size={16} className="text-red" />
                        <span>Decision Precision (BLOCK):</span>
                      </div>
                      <strong className="op-kpi-val font-mono">
                        {(data.operational_metrics.decision_precision_block * 100).toFixed(1)}%
                      </strong>
                      <span className="op-kpi-sub font-mono">
                        ({data.operational_metrics.fraud_in_block_count} /{' '}
                        {data.operational_metrics.total_blocks_count} blocked transactions were actual fraud)
                      </span>
                    </div>

                    <div className="op-kpi-item">
                      <div className="op-kpi-label">
                        <CheckCircle2 size={16} className="text-green" />
                        <span>Decision Recall / Intervention Rate:</span>
                      </div>
                      <strong className="op-kpi-val font-mono">
                        {(data.operational_metrics.decision_recall_intervention * 100).toFixed(1)}%
                      </strong>
                      <span className="op-kpi-sub">
                        Total fraud transactions mitigated by BLOCK or REVIEW
                      </span>
                    </div>

                    {data.pr_auc !== null && (
                      <div className="op-kpi-item">
                        <div className="op-kpi-label">
                          <BarChart2 size={16} className="text-primary" />
                          <span>PR-AUC / ROC-AUC:</span>
                        </div>
                        <strong className="op-kpi-val font-mono">
                          PR-AUC: {data.pr_auc.toFixed(3)} | ROC: {data.roc_auc !== null ? data.roc_auc.toFixed(3) : 'N/A'}
                        </strong>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* 4. Mathematical Degradation Alerts Table */}
              <section aria-label="Degradation Alerts" style={{ marginTop: '1.5rem' }}>
                <div className="section-header-row">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <AlertTriangle size={18} className="text-amber" />
                    <h3 className="section-title-sm">Performance Degradation Diagnostics</h3>
                  </div>
                </div>

                {Object.keys(data.degradation_results).length === 0 ? (
                  <EmptyState
                    title="Zero Performance Degradation"
                    description="All observed classification and operational metrics match or exceed baseline benchmarks."
                    icon={<CheckCircle2 size={24} style={{ color: 'var(--status-approve)' }} />}
                  />
                ) : (
                  <div className="alerts-table-container">
                    <table className="monitoring-table">
                      <thead>
                        <tr>
                          <th>Metric Name</th>
                          <th>Status</th>
                          <th>Observed Value</th>
                          <th>Baseline Benchmark</th>
                          <th>Relative Delta (%)</th>
                          <th>Diagnostic Explanation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(data.degradation_results).map(
                          ([key, item]: [string, MetricDegradationItemResponse]) => (
                            <tr key={key}>
                              <td>
                                <strong className="font-mono">{item.metric_name}</strong>
                              </td>
                              <td>{getSeverityBadge(item.severity)}</td>
                              <td>
                                <span className="font-mono val-highlight">
                                  {(item.observed_value * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td>
                                <span className="font-mono text-muted">
                                  {(item.baseline_value * 100).toFixed(2)}%
                                </span>
                              </td>
                              <td>
                                <span
                                  className={`font-mono ${
                                    item.relative_delta < -0.1 ? 'text-red' : 'text-muted'
                                  }`}
                                >
                                  {(item.relative_delta * 100).toFixed(1)}%
                                </span>
                              </td>
                              <td>
                                <span className="alert-message-text">{item.alert_message || 'Nominal'}</span>
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </>
          )}
        </>
      ) : null}
    </div>
  );
};
