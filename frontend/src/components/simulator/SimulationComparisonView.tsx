import React, { useState } from 'react';
import {
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Minus,
  Scale,
  Layers,
} from 'lucide-react';
import { Badge } from '../common/Badge.tsx';
import type {
  BaselineEvaluationSummary,
  SimulatedEvaluationSummary,
  SimulationComparisonSummary,
  RuleDiffStatus,
} from '../../types/api.ts';

interface SimulationComparisonViewProps {
  baseline: BaselineEvaluationSummary;
  simulated: SimulatedEvaluationSummary;
  comparison: SimulationComparisonSummary;
}

export const SimulationComparisonView: React.FC<SimulationComparisonViewProps> = ({
  baseline,
  simulated,
  comparison,
}) => {
  const [showOnlyModifiedFeatures, setShowOnlyModifiedFeatures] = useState<boolean>(true);
  const [ruleFilter, setRuleFilter] = useState<'ALL' | 'ACTIVE_DIFFS'>('ACTIVE_DIFFS');

  const scoreDelta = comparison.risk_score_delta;
  const isScoreIncreased = scoreDelta > 0;
  const isScoreDecreased = scoreDelta < 0;

  const filteredFeatureDiffs = showOnlyModifiedFeatures
    ? comparison.feature_diffs.filter((f) => f.is_modified)
    : comparison.feature_diffs;

  const filteredRuleDiffs = ruleFilter === 'ACTIVE_DIFFS'
    ? comparison.rule_diffs.filter((r) => r.diff_status !== 'NEITHER')
    : comparison.rule_diffs;

  const getRuleDiffStatusBadge = (status: RuleDiffStatus) => {
    switch (status) {
      case 'NEWLY_TRIGGERED':
        return <Badge variant="block">NEWLY TRIGGERED</Badge>;
      case 'RESOLVED':
        return <Badge variant="approve">RESOLVED (NO LONGER ACTIVE)</Badge>;
      case 'PERSISTENT':
        return <Badge variant="review">PERSISTENT MATCH</Badge>;
      case 'NEITHER':
      default:
        return <Badge variant="neutral">NOT TRIGGERED</Badge>;
    }
  };

  return (
    <div className="simulation-comparison-container">
      {/* 1. High-Level Comparison KPI Cards */}
      <div className="comparison-kpi-grid">
        {/* Risk Score Delta Card */}
        <div className="comparison-kpi-card">
          <div className="comparison-kpi-label">Risk Score Delta</div>
          <div className="comparison-kpi-values">
            <span className="base-score-pill">{baseline.risk_score}</span>
            <ArrowRight size={16} className="transition-arrow" />
            <span className={`sim-score-pill ${isScoreIncreased ? 'increased' : isScoreDecreased ? 'decreased' : ''}`}>
              {simulated.risk_score}
            </span>
            <div className={`delta-tag ${isScoreIncreased ? 'delta-risk-up' : isScoreDecreased ? 'delta-risk-down' : 'delta-neutral'}`}>
              {isScoreIncreased ? (
                <>
                  <TrendingUp size={14} />
                  <span>+{scoreDelta}</span>
                </>
              ) : isScoreDecreased ? (
                <>
                  <TrendingDown size={14} />
                  <span>{scoreDelta}</span>
                </>
              ) : (
                <>
                  <Minus size={14} />
                  <span>0</span>
                </>
              )}
            </div>
          </div>
          <div className="comparison-kpi-sublabel">
            Model Prob: {(baseline.model_score * 100).toFixed(1)}% &rarr; {(simulated.model_score * 100).toFixed(1)}% ({comparison.model_score_delta >= 0 ? '+' : ''}{(comparison.model_score_delta * 100).toFixed(2)}%)
          </div>
        </div>

        {/* Risk Tier Shift Card */}
        <div className="comparison-kpi-card">
          <div className="comparison-kpi-label">Risk Tier Transition</div>
          <div className="comparison-kpi-values">
            <Badge variant={baseline.risk_tier.toLowerCase() as any}>{baseline.risk_tier}</Badge>
            <ArrowRight size={16} className="transition-arrow" />
            <Badge variant={simulated.risk_tier.toLowerCase() as any}>{simulated.risk_tier}</Badge>
          </div>
          <div className="comparison-kpi-sublabel">
            {comparison.tier_changed ? (
              <span style={{ color: 'var(--status-review)' }}>Tier escalated / de-escalated</span>
            ) : (
              <span style={{ color: 'var(--text-muted)' }}>Risk tier unchanged</span>
            )}
          </div>
        </div>

        {/* Decision Policy Action Card */}
        <div className="comparison-kpi-card">
          <div className="comparison-kpi-label">Decision Policy Action</div>
          <div className="comparison-kpi-values">
            <Badge variant={baseline.decision_action.toLowerCase() as any}>{baseline.decision_action}</Badge>
            <ArrowRight size={16} className="transition-arrow" />
            <Badge variant={simulated.decision_action.toLowerCase() as any}>{simulated.decision_action}</Badge>
          </div>
          <div className="comparison-kpi-sublabel">
            {comparison.action_changed ? (
              <span style={{ color: 'var(--status-block)' }}>Action shifted by scenario</span>
            ) : (
              <span style={{ color: 'var(--text-muted)' }}>Policy action maintained</span>
            )}
          </div>
        </div>

        {/* Override Status Card */}
        <div className="comparison-kpi-card">
          <div className="comparison-kpi-label">Rule Override Status</div>
          <div className="comparison-kpi-values">
            {simulated.is_overridden ? (
              <Badge variant="block">RULE OVERRIDE ACTIVE</Badge>
            ) : (
              <Badge variant="neutral">ML POLICY DIRECT</Badge>
            )}
          </div>
          <div className="comparison-kpi-sublabel">
            {comparison.modified_features_count} feature(s) modified
          </div>
        </div>
      </div>

      {/* 2. Rule Impact Comparison Table */}
      <div className="comparison-section">
        <div className="comparison-section-header">
          <div className="section-title-group">
            <Scale size={18} />
            <h3 className="section-title">Business Rule Impact Analysis</h3>
          </div>
          <div className="filter-button-group">
            <button
              type="button"
              className={`filter-btn ${ruleFilter === 'ACTIVE_DIFFS' ? 'active' : ''}`}
              onClick={() => setRuleFilter('ACTIVE_DIFFS')}
            >
              Active / Changed Rules ({comparison.rule_diffs.filter((r) => r.diff_status !== 'NEITHER').length})
            </button>
            <button
              type="button"
              className={`filter-btn ${ruleFilter === 'ALL' ? 'active' : ''}`}
              onClick={() => setRuleFilter('ALL')}
            >
              All Catalog Rules ({comparison.rule_diffs.length})
            </button>
          </div>
        </div>

        {filteredRuleDiffs.length === 0 ? (
          <div className="empty-diff-state">No active rule diffs detected for this simulation.</div>
        ) : (
          <div className="table-responsive">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Rule ID</th>
                  <th>Description</th>
                  <th>Outcome</th>
                  <th>Baseline Status</th>
                  <th>Simulated Status</th>
                  <th>Transition Impact</th>
                </tr>
              </thead>
              <tbody>
                {filteredRuleDiffs.map((rule) => (
                  <tr key={rule.rule_id} className={`rule-diff-row ${rule.diff_status.toLowerCase()}`}>
                    <td className="mono-text font-bold">{rule.rule_id}</td>
                    <td>{rule.description}</td>
                    <td>
                      <Badge variant={rule.outcome.toLowerCase() as any}>{rule.outcome}</Badge>
                    </td>
                    <td>
                      {rule.baseline_triggered ? (
                        <span className="status-triggered font-bold">Triggered</span>
                      ) : (
                        <span className="status-inactive">Passed</span>
                      )}
                    </td>
                    <td>
                      {rule.simulated_triggered ? (
                        <span className="status-triggered font-bold">Triggered</span>
                      ) : (
                        <span className="status-inactive">Passed</span>
                      )}
                    </td>
                    <td>{getRuleDiffStatusBadge(rule.diff_status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 3. Itemized Feature Diff Table */}
      <div className="comparison-section">
        <div className="comparison-section-header">
          <div className="section-title-group">
            <Layers size={18} />
            <h3 className="section-title">Itemized Feature Value Shifts</h3>
          </div>
          <div className="filter-button-group">
            <button
              type="button"
              className={`filter-btn ${showOnlyModifiedFeatures ? 'active' : ''}`}
              onClick={() => setShowOnlyModifiedFeatures(true)}
            >
              Modified Only ({comparison.modified_features_count})
            </button>
            <button
              type="button"
              className={`filter-btn ${!showOnlyModifiedFeatures ? 'active' : ''}`}
              onClick={() => setShowOnlyModifiedFeatures(false)}
            >
              All 55 Features
            </button>
          </div>
        </div>

        {filteredFeatureDiffs.length === 0 ? (
          <div className="empty-diff-state">No feature modifications made against baseline.</div>
        ) : (
          <div className="table-responsive">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Category</th>
                  <th>Baseline Value</th>
                  <th>Simulated Value</th>
                  <th>Delta / Shift</th>
                </tr>
              </thead>
              <tbody>
                {filteredFeatureDiffs.map((f) => (
                  <tr key={f.feature_name} className={f.is_modified ? 'row-modified' : ''}>
                    <td className="font-medium">
                      <div>{f.display_name}</div>
                      <span className="mono-text sublabel-text">{f.feature_name}</span>
                    </td>
                    <td>
                      <span className="category-tag">{f.category}</span>
                    </td>
                    <td className="mono-text">{f.baseline_value !== null && f.baseline_value !== undefined ? String(f.baseline_value) : '—'}</td>
                    <td className="mono-text font-bold">{String(f.simulated_value)}</td>
                    <td>
                      {f.is_modified ? (
                        f.delta !== null && f.delta !== undefined ? (
                          <span className={`diff-delta ${f.delta > 0 ? 'delta-pos' : f.delta < 0 ? 'delta-neg' : ''}`}>
                            {f.delta > 0 ? `+${f.delta}` : f.delta}
                          </span>
                        ) : (
                          <Badge variant="info">VALUE CHANGED</Badge>
                        )
                      ) : (
                        <span className="diff-unchanged">Unchanged</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
