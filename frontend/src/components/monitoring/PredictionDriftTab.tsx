import React, { useEffect, useState, useCallback } from 'react';
import {
  TrendingUp,
  Sliders,
  Gauge,
  SlidersHorizontal,
} from 'lucide-react';
import { fetchPredictionDrift } from '../../api/monitoringApi.ts';
import { DistributionComparisonChart } from './DistributionComparisonChart.tsx';
import { StatCard } from '../common/StatCard.tsx';
import { Badge } from '../common/Badge.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import type {
  MonitoringTimeWindow,
  PredictionDriftResponse,
} from '../../types/monitoring.ts';

interface PredictionDriftTabProps {
  window: MonitoringTimeWindow;
  startTime?: string;
  endTime?: string;
  modelVersion?: string;
}

export const PredictionDriftTab: React.FC<PredictionDriftTabProps> = ({
  window,
  startTime,
  endTime,
  modelVersion = '1.0.0',
}) => {
  const [data, setData] = useState<PredictionDriftResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadPredictionDrift = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchPredictionDrift({
        window,
        start_time: startTime,
        end_time: endTime,
        model_version: modelVersion,
      });
      setData(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction drift report');
    } finally {
      setLoading(false);
    }
  }, [window, startTime, endTime, modelVersion]);

  useEffect(() => {
    loadPredictionDrift();
  }, [loadPredictionDrift]);

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

  const getTierColor = (tier: string) => {
    switch (tier.toUpperCase()) {
      case 'LOW':
        return 'var(--status-approve)';
      case 'MEDIUM':
        return 'var(--status-review)';
      case 'HIGH':
        return 'var(--status-block)';
      case 'CRITICAL':
        return '#ff3366';
      default:
        return 'var(--text-muted)';
    }
  };

  const getActionColor = (action: string) => {
    switch (action.toUpperCase()) {
      case 'APPROVE':
        return 'var(--status-approve)';
      case 'REVIEW':
        return 'var(--status-review)';
      case 'BLOCK':
        return 'var(--status-block)';
      default:
        return 'var(--text-muted)';
    }
  };

  return (
    <div className="monitoring-tab-content">
      {error && (
        <ErrorBanner
          title="Prediction Drift Telemetry Error"
          message={error}
          onRetry={loadPredictionDrift}
        />
      )}

      {loading ? (
        <LoadingSpinner message="Evaluating output probability distributions, risk score buckets, and tier shifts..." />
      ) : data ? (
        <>
          {/* 1. Score & Decision Output KPI Overview */}
          <div className="stats-grid">
            <StatCard
              title="Continuous Model Score PSI"
              value={data.model_score_psi.toFixed(4)}
              icon={<TrendingUp size={18} />}
              sublabel="10 Equal-Width Probability Bins"
              subvalue={`Mean Score: ${data.model_score_mean.toFixed(3)}`}
              tone={
                data.model_score_status === 'CRITICAL'
                  ? 'block'
                  : data.model_score_status === 'WARNING'
                  ? 'review'
                  : 'approve'
              }
              badge={getSeverityBadge(data.model_score_status)}
            />

            <StatCard
              title="Normalized Risk Score PSI"
              value={data.risk_score_psi.toFixed(4)}
              icon={<Gauge size={18} />}
              sublabel="10 Standardized Score Buckets"
              subvalue="0–9 ... 90–100 Scale"
              tone={
                data.risk_score_status === 'CRITICAL'
                  ? 'block'
                  : data.risk_score_status === 'WARNING'
                  ? 'review'
                  : 'approve'
              }
              badge={getSeverityBadge(data.risk_score_status)}
            />

            <StatCard
              title="Risk Tier Stratification JSD"
              value={data.risk_tier_jsd.toFixed(4)}
              icon={<SlidersHorizontal size={18} />}
              sublabel="4 Categorical Tiers"
              subvalue="LOW, MED, HIGH, CRIT"
              tone={
                data.risk_tier_status === 'CRITICAL'
                  ? 'block'
                  : data.risk_tier_status === 'WARNING'
                  ? 'review'
                  : 'approve'
              }
              badge={getSeverityBadge(data.risk_tier_status)}
            />

            <StatCard
              title="Rule Override Frequency"
              value={`${(data.override_rate_current * 100).toFixed(1)}%`}
              icon={<Sliders size={18} />}
              sublabel={`Baseline: ${(data.override_rate_baseline * 100).toFixed(1)}%`}
              subvalue={`&Delta; ${(data.override_rate_delta * 100).toFixed(1)}%`}
              tone={Math.abs(data.override_rate_delta) > 0.05 ? 'review' : 'approve'}
              badge={
                <Badge variant={Math.abs(data.override_rate_delta) > 0.05 ? 'review' : 'neutral'}>
                  {Math.abs(data.override_rate_delta) > 0.05 ? 'SHIFT' : 'STABLE'}
                </Badge>
              }
            />
          </div>

          {/* 2. Distribution Comparison Section */}
          <div className="prediction-charts-grid" style={{ marginTop: '1.5rem' }}>
            {/* Model Probability Distribution */}
            <div className="analytics-card">
              <DistributionComparisonChart
                title="Continuous Model Probability Distribution"
                subtitle="10 equal-width bins across probability spectrum [0.0, 1.0]"
                baselineDistribution={{
                  bin_0: 0.85,
                  bin_1: 0.05,
                  bin_2: 0.03,
                  bin_3: 0.02,
                  bin_4: 0.01,
                  bin_5: 0.01,
                  bin_6: 0.01,
                  bin_7: 0.008,
                  bin_8: 0.007,
                  bin_9: 0.005,
                }}
                currentDistribution={data.model_score_distribution}
                binEdges={[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
                height={200}
              />
            </div>

            {/* Normalized Risk Score 10-Bucket Distribution */}
            <div className="analytics-card">
              <DistributionComparisonChart
                title="Normalized Risk Score Bucket Distribution"
                subtitle="Standardized histogram deciles across integer risk index [0, 100]"
                baselineDistribution={{
                  bucket_0: 0.82,
                  bucket_1: 0.06,
                  bucket_2: 0.03,
                  bucket_3: 0.02,
                  bucket_4: 0.02,
                  bucket_5: 0.01,
                  bucket_6: 0.01,
                  bucket_7: 0.01,
                  bucket_8: 0.01,
                  bucket_9: 0.01,
                }}
                currentDistribution={data.risk_score_buckets_distribution}
                binEdges={[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
                height={200}
              />
            </div>
          </div>

          {/* 3. Categorical Proportions (Risk Tiers & Decision Actions) */}
          <div className="prediction-categorical-grid" style={{ marginTop: '1.5rem' }}>
            {/* Risk Tier Stack */}
            <div className="analytics-card">
              <div className="section-sub-header">
                <div>
                  <h4 className="sub-title">Risk Tier Stratification</h4>
                  <span className="sub-meta">Observed Window vs Tier Categories (JSD: {data.risk_tier_jsd.toFixed(4)})</span>
                </div>
                {getSeverityBadge(data.risk_tier_status)}
              </div>

              <div className="stacked-proportions-bar" style={{ marginTop: '1rem' }}>
                {Object.entries(data.risk_tier_distribution).map(([tier, prop]) => (
                  <div
                    key={tier}
                    className="stack-segment"
                    style={{
                      width: `${Math.max(2, prop * 100)}%`,
                      backgroundColor: getTierColor(tier),
                    }}
                    title={`${tier}: ${(prop * 100).toFixed(1)}%`}
                  />
                ))}
              </div>

              <div className="stack-legend-grid" style={{ marginTop: '0.75rem' }}>
                {Object.entries(data.risk_tier_distribution).map(([tier, prop]) => (
                  <div key={tier} className="stack-legend-item">
                    <span className="legend-dot" style={{ backgroundColor: getTierColor(tier) }} />
                    <span className="legend-name">{tier}:</span>
                    <strong className="legend-val font-mono">{(prop * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </div>

            {/* Decision Action Stack */}
            <div className="analytics-card">
              <div className="section-sub-header">
                <div>
                  <h4 className="sub-title">Operational Decision Action Distribution</h4>
                  <span className="sub-meta">Enforced Actions Breakdown (JSD: {data.action_jsd.toFixed(4)})</span>
                </div>
                {getSeverityBadge(data.action_status)}
              </div>

              <div className="stacked-proportions-bar" style={{ marginTop: '1rem' }}>
                {Object.entries(data.action_distribution).map(([act, prop]) => (
                  <div
                    key={act}
                    className="stack-segment"
                    style={{
                      width: `${Math.max(2, prop * 100)}%`,
                      backgroundColor: getActionColor(act),
                    }}
                    title={`${act}: ${(prop * 100).toFixed(1)}%`}
                  />
                ))}
              </div>

              <div className="stack-legend-grid" style={{ marginTop: '0.75rem' }}>
                {Object.entries(data.action_distribution).map(([act, prop]) => (
                  <div key={act} className="stack-legend-item">
                    <span className="legend-dot" style={{ backgroundColor: getActionColor(act) }} />
                    <span className="legend-name">{act}:</span>
                    <strong className="legend-val font-mono">{(prop * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
