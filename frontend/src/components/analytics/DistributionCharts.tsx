import React, { useState } from 'react';
import { BarChart3 } from 'lucide-react';
import type {
  CategoryCount,
  DistributionBucket,
} from '../../types/api.ts';

interface DistributionChartsProps {
  totalEvaluated: number;
  riskScoreDistribution: DistributionBucket[];
  modelScoreDistribution: DistributionBucket[];
  riskTierDistribution: CategoryCount[];
  decisionDistribution: CategoryCount[];
}

export const DistributionCharts: React.FC<DistributionChartsProps> = ({
  totalEvaluated,
  riskScoreDistribution,
  modelScoreDistribution,
  riskTierDistribution,
  decisionDistribution,
}) => {
  const [activeScoreTab, setActiveScoreTab] = useState<'risk_score' | 'model_score'>('risk_score');
  const [hoveredBucketIndex, setHoveredBucketIndex] = useState<number | null>(null);

  const activeBuckets = activeScoreTab === 'risk_score' ? riskScoreDistribution : modelScoreDistribution;
  const maxBucketCount = Math.max(...activeBuckets.map((b) => b.count), 1);

  const getBucketColor = (_bucketLabel: string, index: number): string => {
    if (activeScoreTab === 'risk_score') {
      if (index <= 2) return 'var(--status-approve)'; // 0-9, 10-19, 20-29 (Low)
      if (index <= 6) return 'var(--status-review)';  // 30-39, 40-49, 50-59, 60-69 (Medium)
      if (index <= 8) return 'var(--status-block)';   // 70-79, 80-89 (High)
      return '#ff3366';                               // 90-100 (Critical)
    }
    // Model score gradient from safe to risky
    if (index <= 2) return 'var(--status-approve)';
    if (index <= 6) return 'var(--status-review)';
    return 'var(--status-block)';
  };

  const getTierColor = (tier: string): string => {
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

  const getDecisionColor = (decision: string): string => {
    switch (decision.toUpperCase()) {
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
    <div className="analytics-distributions-grid">
      {/* 1. Score Distribution Histogram */}
      <div className="analytics-card histogram-card">
        <div className="chart-header">
          <div>
            <div className="chart-title-group">
              <BarChart3 size={16} />
              <h3 className="chart-title">Risk Score &amp; Probability Distribution</h3>
            </div>
            <span className="chart-subtitle">
              Aggregated frequency distribution across 10 calibrated risk intervals ({totalEvaluated.toLocaleString()} evaluations)
            </span>
          </div>

          <div className="tab-pill-group">
            <button
              type="button"
              className={`tab-pill ${activeScoreTab === 'risk_score' ? 'active' : ''}`}
              onClick={() => {
                setActiveScoreTab('risk_score');
                setHoveredBucketIndex(null);
              }}
            >
              Calibrated Score (0–100)
            </button>
            <button
              type="button"
              className={`tab-pill ${activeScoreTab === 'model_score' ? 'active' : ''}`}
              onClick={() => {
                setActiveScoreTab('model_score');
                setHoveredBucketIndex(null);
              }}
            >
              Model Margin P(0.0–1.0)
            </button>
          </div>
        </div>

        {/* Histogram Bars */}
        <div className="histogram-bars-container">
          {activeBuckets.map((bucket, idx) => {
            const heightPct = Math.max(4, (bucket.count / maxBucketCount) * 100);
            const color = getBucketColor(bucket.bucket_label, idx);
            const isHovered = hoveredBucketIndex === idx;

            return (
              <div
                key={bucket.bucket_label}
                className={`histogram-column ${isHovered ? 'hovered' : ''}`}
                onMouseEnter={() => setHoveredBucketIndex(idx)}
                onMouseLeave={() => setHoveredBucketIndex(null)}
                tabIndex={0}
                role="graphics-symbol"
                aria-label={`Bucket ${bucket.bucket_label}: ${bucket.count} transactions (${bucket.percentage.toFixed(1)}%)`}
              >
                <div className="bar-wrapper">
                  <div
                    className="histogram-bar"
                    style={{
                      height: `${heightPct}%`,
                      backgroundColor: color,
                      opacity: isHovered ? 1.0 : 0.85,
                      boxShadow: isHovered ? `0 0 12px ${color}` : 'none',
                    }}
                  >
                    <span className="bar-count-label font-mono">
                      {bucket.count > 0 ? bucket.count : ''}
                    </span>
                  </div>
                </div>

                <div className="bar-bucket-label font-mono">
                  {bucket.bucket_label}
                </div>
                <div className="bar-pct-label font-mono">
                  {bucket.percentage.toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>

        {/* Active Histogram Bucket Detail Card */}
        {hoveredBucketIndex !== null && activeBuckets[hoveredBucketIndex] && (
          <div className="histogram-detail-banner">
            <div className="banner-item">
              <span className="b-label">Score Range:</span>
              <strong className="font-mono">{activeBuckets[hoveredBucketIndex].bucket_label}</strong>
            </div>
            <div className="banner-item">
              <span className="b-label">Evaluated Count:</span>
              <strong className="font-mono">{activeBuckets[hoveredBucketIndex].count.toLocaleString()} transactions</strong>
            </div>
            <div className="banner-item">
              <span className="b-label">Proportion:</span>
              <strong className="font-mono">{activeBuckets[hoveredBucketIndex].percentage.toFixed(2)}% of dataset</strong>
            </div>
          </div>
        )}

        <div className="histogram-legend">
          <div className="legend-item">
            <span className="legend-swatch approve-swatch" />
            <span>Low Risk (0–29)</span>
          </div>
          <div className="legend-item">
            <span className="legend-swatch review-swatch" />
            <span>Medium Risk (30–69)</span>
          </div>
          <div className="legend-item">
            <span className="legend-swatch block-swatch" />
            <span>High Risk (70–89)</span>
          </div>
          <div className="legend-item">
            <span className="legend-swatch" style={{ background: '#ff3366' }} />
            <span>Critical Risk (90–100)</span>
          </div>
        </div>
      </div>

      {/* 2. Categorical Distributions (Risk Tiers & Decision Actions) */}
      <div className="analytics-card categorical-distribution-card">
        {/* Tier Distribution Section */}
        <div className="categorical-section">
          <div className="section-sub-header">
            <h4 className="sub-title">Risk Tier Stratification</h4>
            <span className="sub-meta">{totalEvaluated.toLocaleString()} Total</span>
          </div>

          <div className="distribution-progress-stack">
            {riskTierDistribution.map((item) => (
              <div
                key={item.category}
                className="progress-segment"
                style={{
                  width: `${Math.max(1, item.percentage)}%`,
                  backgroundColor: getTierColor(item.category),
                }}
                title={`${item.category}: ${item.count} (${item.percentage.toFixed(1)}%)`}
              />
            ))}
          </div>

          <div className="distribution-items-list">
            {riskTierDistribution.map((item) => (
              <div key={item.category} className="dist-item-row">
                <div className="dist-item-label">
                  <span
                    className="dist-dot"
                    style={{ backgroundColor: getTierColor(item.category) }}
                  />
                  <span className="font-semibold">{item.category} RISK</span>
                </div>
                <div className="dist-item-vals font-mono">
                  <span className="dist-cnt">{item.count.toLocaleString()}</span>
                  <span className="dist-pct">({item.percentage.toFixed(1)}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Decision Distribution Section */}
        <div className="categorical-section">
          <div className="section-sub-header">
            <h4 className="sub-title">Operational Decision Distribution</h4>
            <span className="sub-meta">{totalEvaluated.toLocaleString()} Total</span>
          </div>

          <div className="distribution-progress-stack">
            {decisionDistribution.map((item) => (
              <div
                key={item.category}
                className="progress-segment"
                style={{
                  width: `${Math.max(1, item.percentage)}%`,
                  backgroundColor: getDecisionColor(item.category),
                }}
                title={`${item.category}: ${item.count} (${item.percentage.toFixed(1)}%)`}
              />
            ))}
          </div>

          <div className="distribution-items-list">
            {decisionDistribution.map((item) => (
              <div key={item.category} className="dist-item-row">
                <div className="dist-item-label">
                  <span
                    className="dist-dot"
                    style={{ backgroundColor: getDecisionColor(item.category) }}
                  />
                  <span className="font-semibold">{item.category} ACTION</span>
                </div>
                <div className="dist-item-vals font-mono">
                  <span className="dist-cnt">{item.count.toLocaleString()}</span>
                  <span className="dist-pct">({item.percentage.toFixed(1)}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
