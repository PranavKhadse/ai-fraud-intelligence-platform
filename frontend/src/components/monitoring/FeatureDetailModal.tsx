import React, { useEffect, useState } from 'react';
import { X, Activity, AlertTriangle, CheckCircle2, ShieldAlert, BarChart2 } from 'lucide-react';
import { fetchFeatureDriftDetail } from '../../api/monitoringApi.ts';
import { DistributionComparisonChart } from './DistributionComparisonChart.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import type { FeatureDriftDetailResponse, MonitoringTimeWindow } from '../../types/monitoring.ts';

interface FeatureDetailModalProps {
  featureName: string;
  timeWindow: MonitoringTimeWindow;
  startTime?: string;
  endTime?: string;
  modelVersion?: string;
  onClose: () => void;
}

export const FeatureDetailModal: React.FC<FeatureDetailModalProps> = ({
  featureName,
  timeWindow,
  startTime,
  endTime,
  modelVersion = '1.0.0',
  onClose,
}) => {
  const [detail, setDetail] = useState<FeatureDriftDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchFeatureDriftDetail(featureName, {
      window: timeWindow,
      start_time: startTime,
      end_time: endTime,
      model_version: modelVersion,
    })
      .then((data) => {
        if (isMounted) {
          setDetail(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to load feature drift detail');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [featureName, timeWindow, startTime, endTime, modelVersion]);

  // Handle ESC key to close modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const getStatusIcon = (status: string) => {
    switch (status.toUpperCase()) {
      case 'CRITICAL':
        return <ShieldAlert size={18} style={{ color: 'var(--status-block)' }} />;
      case 'WARNING':
        return <AlertTriangle size={18} style={{ color: 'var(--status-review)' }} />;
      case 'NORMAL':
        return <CheckCircle2 size={18} style={{ color: 'var(--status-approve)' }} />;
      default:
        return <Activity size={18} style={{ color: 'var(--text-muted)' }} />;
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-card feature-detail-modal" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title-group">
            <BarChart2 size={20} className="text-primary" />
            <div>
              <h3 className="modal-title font-mono">{featureName}</h3>
              <span className="modal-subtitle">
                Single-Feature Drift Diagnostics &amp; Baseline Comparison
              </span>
            </div>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {loading ? (
            <LoadingSpinner message={`Evaluating baseline profile vs current window for ${featureName}...`} />
          ) : error ? (
            <ErrorBanner title="Diagnostic Inspection Failed" message={error} onRetry={() => {}} />
          ) : detail ? (
            <div className="feature-detail-content">
              {/* Header Status & Metadata Row */}
              <div className="feature-meta-bar">
                <div className="meta-badge-group">
                  <span className={`status-badge status-${detail.status.toLowerCase()}`}>
                    {getStatusIcon(detail.status)}
                    <span>{detail.status}</span>
                  </span>
                  <span className="type-badge font-mono">{detail.feature_type.toUpperCase()}</span>
                  <span className="category-badge">{detail.category}</span>
                </div>
                <div className="window-tag font-mono">
                  Window: {timeWindow} (v{modelVersion})
                </div>
              </div>

              {/* Statistical Metrics Summary Grid */}
              <div className="feature-metric-cards-grid">
                <div className="diag-stat-card">
                  <span className="diag-stat-label">Population Stability Index (PSI)</span>
                  <strong className="diag-stat-val font-mono">{detail.psi.toFixed(4)}</strong>
                  <span className="diag-stat-sub">Threshold: &ge;0.10 (Warn), &ge;0.25 (Crit)</span>
                </div>

                {detail.feature_type === 'numerical' ? (
                  <>
                    <div className="diag-stat-card">
                      <span className="diag-stat-label">Two-Sample KS Statistic</span>
                      <strong className="diag-stat-val font-mono">
                        {detail.ks_statistic !== null ? detail.ks_statistic.toFixed(4) : 'N/A'}
                      </strong>
                      <span className="diag-stat-sub">Max CDF Deviation</span>
                    </div>

                    <div className="diag-stat-card">
                      <span className="diag-stat-label">KS Asymptotic p-value</span>
                      <strong className="diag-stat-val font-mono">
                        {detail.ks_p_value !== null
                          ? detail.ks_p_value < 0.001
                            ? detail.ks_p_value.toExponential(2)
                            : detail.ks_p_value.toFixed(4)
                          : 'N/A'}
                      </strong>
                      <span className="diag-stat-sub">Critical: &le;0.01</span>
                    </div>
                  </>
                ) : (
                  <div className="diag-stat-card">
                    <span className="diag-stat-label">Jensen-Shannon Divergence</span>
                    <strong className="diag-stat-val font-mono">
                      {detail.js_divergence !== null ? detail.js_divergence.toFixed(4) : 'N/A'}
                    </strong>
                    <span className="diag-stat-sub">Categorical shift metric</span>
                  </div>
                )}

                <div className="diag-stat-card">
                  <span className="diag-stat-label">Missing Rate Delta</span>
                  <strong
                    className={`diag-stat-val font-mono ${
                      Math.abs(detail.missing_rate_delta) > 0.05 ? 'text-amber' : ''
                    }`}
                  >
                    {(detail.missing_rate_delta * 100).toFixed(2)}%
                  </strong>
                  <span className="diag-stat-sub">
                    Base: {(detail.missing_rate_baseline * 100).toFixed(1)}% | Curr:{' '}
                    {(detail.missing_rate_current * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Unseen Categories Section if present */}
              {detail.unseen_categories && detail.unseen_categories.length > 0 && (
                <div className="unseen-categories-banner">
                  <AlertTriangle size={16} className="text-amber" />
                  <div>
                    <strong>{detail.unseen_categories.length} New Unseen Categories Encountered:</strong>{' '}
                    <span className="font-mono">
                      {detail.unseen_categories.slice(0, 10).join(', ')}
                      {detail.unseen_categories.length > 10 ? ' ...' : ''}
                    </span>
                  </div>
                </div>
              )}

              {/* Distribution Comparison Chart */}
              <div className="feature-dist-chart-section">
                <DistributionComparisonChart
                  title="Empirical Decile & Category Proportions"
                  subtitle="Comparison of reference baseline profile distribution against observed transactions"
                  baselineDistribution={detail.baseline_distribution}
                  currentDistribution={detail.current_distribution}
                  binEdges={detail.bin_edges}
                  height={200}
                />
              </div>
            </div>
          ) : null}
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Close Diagnostic View
          </button>
        </div>
      </div>
    </div>
  );
};
