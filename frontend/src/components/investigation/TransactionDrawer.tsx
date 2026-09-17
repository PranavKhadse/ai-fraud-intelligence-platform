import React, { useEffect, useState, useCallback } from 'react';
import {
  X,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Clock,
  MapPin,
  DollarSign,
  Layers,
  FileText,
  Activity,
  ArrowRightLeft,
  ChevronDown,
  ChevronRight,
  Sliders,
  TrendingUp,
  TrendingDown,
  Scale,
} from 'lucide-react';
import { Badge } from '../common/Badge.tsx';
import { RiskScoreMeter } from '../common/RiskScoreMeter.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { ShapWaterfall } from '../explainability/ShapWaterfall.tsx';
import { fetchTransactionDetail } from '../../api/dashboardApi.ts';
import type {
  DecisionAction,
  RiskTier,
  RuleOutcome,
  AttributionDirection,
  ReasonSeverity,
  TransactionDetailResponse,
} from '../../types/api.ts';

interface TransactionDrawerProps {
  transactionId: string | null;
  onClose: () => void;
  onSimulateTransaction?: (detail: TransactionDetailResponse) => void;
}

// 7 Feature Category Definition Mappings from Phase 3 Feature Engine
const FEATURE_CATEGORY_MAP: Record<string, string[]> = {
  'Velocity & Frequency': [
    'txn_count_1h',
    'txn_count_6h',
    'txn_count_24h',
    'txn_count_7d',
    'txn_count_30d',
    'time_since_prev_txn_seconds',
    'is_first_account_txn',
  ],
  'Spending & Monetary Volume': [
    'amt_sum_1h',
    'amt_sum_24h',
    'amt_sum_7d',
    'amt_sum_30d',
    'amt_mean_24h',
    'amt_mean_7d',
    'amt_max_24h',
    'amt_median_30d',
  ],
  'Spending Deviation & Z-Scores': [
    'historical_amount_mean',
    'historical_amount_std',
    'historical_amount_median',
    'amount_zscore',
    'amount_ratio_to_historical_mean',
  ],
  'Account History & Diversity': [
    'account_txn_count_before',
    'account_total_spend_before',
    'account_avg_amount_before',
    'account_max_amount_before',
    'account_unique_merchant_count_before',
    'account_unique_category_count_before',
  ],
  'Merchant & Category Interactions': [
    'account_merchant_txn_count_before',
    'account_category_txn_count_before',
    'account_merchant_spend_before',
    'account_category_spend_before',
    'merchant_txn_count_before',
    'category_txn_count_before',
  ],
  'Geographic & Impossible Travel': [
    'cardholder_merchant_distance_km',
    'distance_from_prev_merchant_km',
    'implied_travel_speed_kmh',
    'is_impossible_travel_speed',
  ],
  'Temporal & Periodicity': [
    'transaction_hour',
    'day_of_week',
    'day_of_month',
    'month',
    'week_of_year',
    'is_weekend',
    'is_night',
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
  ],
};

export const TransactionDrawer: React.FC<TransactionDrawerProps> = ({
  transactionId,
  onClose,
  onSimulateTransaction,
}) => {
  const [data, setData] = useState<TransactionDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({
    'Velocity & Frequency': true,
    'Spending & Monetary Volume': true,
    'Spending Deviation & Z-Scores': true,
    'Geographic & Impossible Travel': true,
  });

  const loadDetail = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetchTransactionDetail(id);
      setData(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to retrieve transaction details.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (transactionId) {
      loadDetail(transactionId);
    } else {
      setData(null);
      setError(null);
    }
  }, [transactionId, loadDetail]);

  // Keyboard accessibility: ESC key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!transactionId) {
    return null;
  }

  const toggleCategory = (catName: string) => {
    setExpandedCategories((prev) => ({
      ...prev,
      [catName]: !prev[catName],
    }));
  };

  const formatCurrency = (val: number, currency: string = 'USD'): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency || 'USD',
      minimumFractionDigits: 2,
    }).format(val);
  };

  const formatTimestamp = (isoString?: string | null): string => {
    if (!isoString) return '—';
    try {
      return new Date(isoString).toLocaleString('en-US', {
        dateStyle: 'medium',
        timeStyle: 'medium',
        hour12: false,
      });
    } catch {
      return isoString;
    }
  };

  const formatValue = (val: unknown): string => {
    if (val === null || val === undefined) return 'null';
    if (typeof val === 'number') {
      return Number.isInteger(val) ? val.toString() : val.toFixed(4);
    }
    if (typeof val === 'boolean') {
      return val ? 'True' : 'False';
    }
    return String(val);
  };

  const renderDecisionBadge = (action: DecisionAction) => {
    switch (action) {
      case 'APPROVE':
        return (
          <Badge variant="approve" icon={<CheckCircle2 size={12} />}>
            APPROVE
          </Badge>
        );
      case 'REVIEW':
        return (
          <Badge variant="review" icon={<AlertTriangle size={12} />}>
            REVIEW
          </Badge>
        );
      case 'BLOCK':
        return (
          <Badge variant="block" icon={<ShieldAlert size={12} />}>
            BLOCK
          </Badge>
        );
    }
  };

  const renderTierBadge = (tier: RiskTier) => {
    switch (tier) {
      case 'LOW':
        return <Badge variant="approve">LOW RISK</Badge>;
      case 'MEDIUM':
        return <Badge variant="review">MEDIUM RISK</Badge>;
      case 'HIGH':
        return <Badge variant="block">HIGH RISK</Badge>;
      case 'CRITICAL':
        return <Badge variant="block">CRITICAL RISK</Badge>;
    }
  };

  const renderOutcomeBadge = (outcome: RuleOutcome) => {
    switch (outcome) {
      case 'APPROVE':
        return <Badge variant="approve">APPROVE</Badge>;
      case 'REVIEW':
        return <Badge variant="review">REVIEW</Badge>;
      case 'BLOCK':
        return <Badge variant="block">BLOCK</Badge>;
      case 'MONITOR':
        return <Badge variant="neutral">MONITOR</Badge>;
    }
  };

  const renderSeverityBadge = (severity: ReasonSeverity) => {
    switch (severity) {
      case 'CRITICAL':
        return <Badge variant="block">CRITICAL</Badge>;
      case 'HIGH':
        return <Badge variant="block">HIGH</Badge>;
      case 'MEDIUM':
        return <Badge variant="review">MEDIUM</Badge>;
      case 'LOW':
        return <Badge variant="approve">LOW</Badge>;
      case 'INFO':
        return <Badge variant="neutral">INFO</Badge>;
    }
  };

  const renderDirectionBadge = (dir: AttributionDirection) => {
    if (dir === 'RISK_INCREASING') {
      return (
        <Badge variant="block" icon={<TrendingUp size={12} />}>
          RISK INCREASING
        </Badge>
      );
    }
    return (
      <Badge variant="approve" icon={<TrendingDown size={12} />}>
        MITIGATING
      </Badge>
    );
  };

  return (
    <div
      className="drawer-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="drawer-title"
    >
      <div
        className="drawer-container"
        onClick={(e) => e.stopPropagation()}
        tabIndex={-1}
      >
        {/* Drawer Header */}
        <div className="drawer-header">
          <div className="drawer-header-info">
            <div className="drawer-header-title-row">
              <h2 id="drawer-title" className="drawer-title">
                Transaction Investigation
              </h2>
              {data?.evaluation && renderDecisionBadge(data.evaluation.decision_action)}
              {data?.evaluation && renderTierBadge(data.evaluation.risk_tier)}
            </div>
            <div className="drawer-header-ids">
              <span className="id-tag">
                UUID: <span className="font-mono">{transactionId}</span>
              </span>
              {data?.transaction?.external_transaction_id && (
                <span className="id-tag">
                  External ID: <span className="font-mono">{data.transaction.external_transaction_id}</span>
                </span>
              )}
            </div>
          </div>

          <div className="drawer-header-actions-group">
            {data && onSimulateTransaction && (
              <button
                type="button"
                className="btn-simulate-drawer"
                onClick={() => {
                  onSimulateTransaction(data);
                  onClose();
                }}
                title="Open in What-If Simulator with this transaction as baseline"
              >
                <Sliders size={14} />
                <span>Simulate in What-If</span>
              </button>
            )}

            <button
              type="button"
              className="drawer-close-btn"
              onClick={onClose}
              aria-label="Close investigation drawer"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Drawer Content Body */}
        <div className="drawer-body">
          {loading && (
            <div className="drawer-loading-wrap">
              <LoadingSpinner message="Retrieving persisted transaction evaluation context..." />
            </div>
          )}

          {error && (
            <div className="drawer-error-wrap">
              <ErrorBanner
                title="Investigation Lookup Failed"
                message={error}
                onRetry={() => loadDetail(transactionId)}
              />
            </div>
          )}

          {!loading && !error && data && (
            <>
              {/* Section 1: Risk Decision Summary */}
              <div className="drawer-section">
                <div className="section-title">
                  <Activity size={16} />
                  <span>Risk Decision & Telemetry</span>
                </div>

                {data.evaluation ? (
                  <div className="risk-assessment-card">
                    <div className="risk-assessment-grid">
                      <div className="assessment-item score-item">
                        <span className="item-label">Calibrated Risk Score</span>
                        <div className="score-meter-wrap">
                          <RiskScoreMeter
                            score={data.evaluation.risk_score}
                            tier={data.evaluation.risk_tier}
                          />
                        </div>
                      </div>

                      <div className="assessment-item">
                        <span className="item-label">Model Probability</span>
                        <span className="item-value font-mono">
                          {(data.evaluation.model_score * 100).toFixed(2)}%{' '}
                          <small className="text-muted">({data.evaluation.model_score.toFixed(6)})</small>
                        </span>
                      </div>

                      <div className="assessment-item">
                        <span className="item-label">Policy Decision</span>
                        <div className="badge-row">
                          {renderDecisionBadge(data.evaluation.decision_action)}
                          {data.evaluation.is_overridden && (
                            <Badge variant="review" icon={<ArrowRightLeft size={11} />}>
                              RULE OVERRIDE
                            </Badge>
                          )}
                        </div>
                      </div>

                      <div className="assessment-item">
                        <span className="item-label">Baseline Action</span>
                        <span className="item-value text-muted font-mono">
                          {data.evaluation.baseline_action}
                        </span>
                      </div>

                      <div className="assessment-item">
                        <span className="item-label">Evaluation Latency</span>
                        <span className="item-value font-mono">
                          {data.evaluation.evaluation_latency_ms !== null
                            ? `${data.evaluation.evaluation_latency_ms.toFixed(2)} ms`
                            : '—'}
                        </span>
                      </div>

                      <div className="assessment-item">
                        <span className="item-label">Champion Model</span>
                        <span className="item-value font-mono">
                          v{data.evaluation.model_version} ({data.evaluation.policy_mode})
                        </span>
                      </div>
                    </div>

                    <div className="decision-reason-box">
                      <strong>Decision Boundary Explanation:</strong>
                      <p>{data.evaluation.decision_reason}</p>
                    </div>
                  </div>
                ) : (
                  <div className="empty-sub-section">
                    <AlertTriangle size={16} />
                    <span>No risk evaluation record is associated with this transaction record.</span>
                  </div>
                )}
              </div>

              {/* Section 2: Canonical Transaction Metadata */}
              <div className="drawer-section">
                <div className="section-title">
                  <DollarSign size={16} />
                  <span>Transaction Metadata</span>
                </div>

                <div className="metadata-grid">
                  <div className="metadata-card">
                    <span className="meta-label">Amount & Currency</span>
                    <span className="meta-val highlight">
                      {formatCurrency(data.transaction.amount, data.transaction.currency)}
                    </span>
                  </div>

                  <div className="metadata-card">
                    <span className="meta-label">Merchant Category</span>
                    <span className="meta-val">{data.transaction.merchant_category}</span>
                  </div>

                  <div className="metadata-card">
                    <span className="meta-label">Cardholder Account ID</span>
                    <span className="meta-val font-mono">{data.transaction.account_id}</span>
                  </div>

                  <div className="metadata-card">
                    <span className="meta-label">Job Classification</span>
                    <span className="meta-val">{data.transaction.job_category}</span>
                  </div>

                  <div className="metadata-card">
                    <span className="meta-label">Cardholder Location</span>
                    <span className="meta-val font-mono text-xs">
                      <MapPin size={11} /> {data.transaction.cardholder_lat.toFixed(4)},{' '}
                      {data.transaction.cardholder_long.toFixed(4)}
                    </span>
                  </div>

                  <div className="metadata-card">
                    <span className="meta-label">Merchant Terminal Location</span>
                    <span className="meta-val font-mono text-xs">
                      <MapPin size={11} /> {data.transaction.merchant_lat.toFixed(4)},{' '}
                      {data.transaction.merchant_long.toFixed(4)}
                    </span>
                  </div>

                  <div className="metadata-card">
                    <span className="meta-label">City Population</span>
                    <span className="meta-val font-mono">
                      {data.transaction.city_pop.toLocaleString()}
                    </span>
                  </div>

                  <div className="metadata-card">
                    <span className="meta-label">Transaction Timestamp</span>
                    <span className="meta-val font-mono text-xs">
                      <Clock size={11} /> {formatTimestamp(data.transaction.transaction_timestamp)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Section 3: TreeSHAP Feature Attributions */}
              <div className="drawer-section">
                <div className="section-title">
                  <Scale size={16} />
                  <span>TreeSHAP Local Explainability</span>
                  <span className="section-badge">{data.feature_attributions.length} Drivers</span>
                </div>

                {data.evaluation && data.feature_attributions.length > 0 && (
                  <ShapWaterfall
                    baseValue={data.evaluation.base_value}
                    outputMargin={data.evaluation.output_margin}
                    modelScore={data.evaluation.model_score}
                    featureAttributions={data.feature_attributions}
                  />
                )}

                {data.feature_attributions.length > 0 ? (
                  <div className="shap-table-wrapper">
                    <table className="shap-table">
                      <thead>
                        <tr>
                          <th>Rank</th>
                          <th>Feature</th>
                          <th>Raw Input</th>
                          <th>SHAP (Log-Odds)</th>
                          <th>Direction</th>
                          <th>Contribution</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.feature_attributions.map((fa) => (
                          <tr key={fa.id}>
                            <td className="cell-rank">#{fa.rank}</td>
                            <td className="cell-feature">
                              <span className="display-name">{fa.display_name}</span>
                              <span className="tech-name font-mono">{fa.feature_name}</span>
                            </td>
                            <td className="cell-raw font-mono">{formatValue(fa.raw_value)}</td>
                            <td
                              className={`cell-shap font-mono ${
                                fa.shap_value >= 0 ? 'shap-positive' : 'shap-negative'
                              }`}
                            >
                              {fa.shap_value >= 0 ? `+${fa.shap_value.toFixed(4)}` : fa.shap_value.toFixed(4)}
                            </td>
                            <td className="cell-direction">{renderDirectionBadge(fa.direction)}</td>
                            <td className="cell-pct font-mono">
                              {fa.relative_contribution_pct.toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="empty-sub-section">
                    <span>No local TreeSHAP feature attributions persisted for this evaluation.</span>
                  </div>
                )}
              </div>

              {/* Section 4: Plain-English Reason Codes */}
              <div className="drawer-section">
                <div className="section-title">
                  <FileText size={16} />
                  <span>Plain-English Reason Codes</span>
                  <span className="section-badge">{data.reason_codes.length} Codes</span>
                </div>

                {data.reason_codes.length > 0 ? (
                  <div className="reason-cards-list">
                    {data.reason_codes.map((rc) => (
                      <div key={rc.id} className="reason-code-card">
                        <div className="reason-card-header">
                          <div className="reason-headline-group">
                            <span className="reason-rank">#{rc.rank}</span>
                            <span className="reason-headline">{rc.headline}</span>
                          </div>
                          <div className="reason-badges">
                            <Badge variant="neutral">{rc.category}</Badge>
                            <Badge variant="neutral">{rc.source}</Badge>
                            {renderSeverityBadge(rc.severity)}
                          </div>
                        </div>
                        <p className="reason-description">{rc.description}</p>
                        <div className="reason-code-tag">
                          Code: <span className="font-mono">{rc.code}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-sub-section">
                    <span>No reason codes were generated for this transaction.</span>
                  </div>
                )}
              </div>

              {/* Section 5: Triggered Business Rules */}
              <div className="drawer-section">
                <div className="section-title">
                  <Sliders size={16} />
                  <span>Triggered Business Rules</span>
                  <span className="section-badge">{data.rule_matches.length} Triggered</span>
                </div>

                {data.rule_matches.length > 0 ? (
                  <div className="rule-cards-list">
                    {data.rule_matches.map((rm) => (
                      <div key={rm.id} className="rule-match-card">
                        <div className="rule-header">
                          <div className="rule-id-wrap">
                            <span className="rule-id font-mono">{rm.rule_id}</span>
                            <span className="rule-priority">Priority #{rm.priority}</span>
                          </div>
                          <div className="rule-outcome-wrap">
                            {renderOutcomeBadge(rm.outcome)}
                            <Badge variant="neutral">{rm.rule_type}</Badge>
                          </div>
                        </div>
                        <p className="rule-desc">{rm.description}</p>
                        <div className="rule-condition font-mono">
                          Condition: <code>{rm.feature_name} {rm.operator} {rm.comparison_value}</code>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-sub-section">
                    <span>No deterministic business rules were triggered during evaluation.</span>
                  </div>
                )}
              </div>

              {/* Section 6: 55-Feature Categorized Snapshot */}
              <div className="drawer-section">
                <div className="section-title">
                  <Layers size={16} />
                  <span>55-Feature Snapshot (Point-in-Time)</span>
                  <span className="section-badge">{Object.keys(data.features).length} Features</span>
                </div>

                <div className="feature-groups-accordion">
                  {Object.entries(FEATURE_CATEGORY_MAP).map(([catTitle, featureNames]) => {
                    const groupFeatures = featureNames.filter((fn) => fn in data.features);
                    if (groupFeatures.length === 0) return null;
                    const isExpanded = expandedCategories[catTitle] ?? false;

                    return (
                      <div key={catTitle} className="feature-group-container">
                        <button
                          type="button"
                          className="feature-group-header"
                          onClick={() => toggleCategory(catTitle)}
                          aria-expanded={isExpanded}
                        >
                          <div className="group-header-left">
                            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                            <span className="group-title">{catTitle}</span>
                          </div>
                          <span className="group-count font-mono text-muted">
                            {groupFeatures.length} features
                          </span>
                        </button>

                        {isExpanded && (
                          <div className="feature-group-body">
                            <div className="feature-grid">
                              {groupFeatures.map((fn) => (
                                <div key={fn} className="feature-pill">
                                  <span className="feat-name font-mono">{fn}</span>
                                  <span className="feat-val font-mono">
                                    {formatValue(data.features[fn])}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Section 7: Audit Trail Timeline */}
              <div className="drawer-section">
                <div className="section-title">
                  <Clock size={16} />
                  <span>Audit Trail Timeline</span>
                  <span className="section-badge">{data.audit_trail.length} Events</span>
                </div>

                {data.audit_trail.length > 0 ? (
                  <div className="audit-timeline">
                    {data.audit_trail.map((log) => (
                      <div key={log.id} className="audit-timeline-item">
                        <div className="timeline-marker" />
                        <div className="timeline-content">
                          <div className="timeline-header">
                            <span className="audit-event-type font-mono">{log.event_type}</span>
                            <span className="audit-action-badge">{log.action}</span>
                            <span className="audit-time font-mono">
                              {formatTimestamp(log.event_timestamp)}
                            </span>
                          </div>
                          <div className="timeline-details">
                            <span className="audit-actor">
                              Actor: <strong>{log.actor_type}</strong> ({log.actor_id || 'system'})
                            </span>
                            {log.correlation_id && (
                              <span className="audit-corr font-mono">
                                Corr: {log.correlation_id}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-sub-section">
                    <span>No audit events associated with this transaction.</span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
