import React from 'react';
import {
  Sliders,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  ArrowRightLeft,
} from 'lucide-react';
import { Badge } from '../common/Badge.tsx';
import type { RuleAnalyticsItem, RuleOutcome } from '../../types/api.ts';

interface RuleAnalyticsTableProps {
  rules: RuleAnalyticsItem[];
  totalEvaluationsAnalyzed: number;
}

export const RuleAnalyticsTable: React.FC<RuleAnalyticsTableProps> = ({
  rules,
  totalEvaluationsAnalyzed,
}) => {
  const renderOutcomeBadge = (outcome: RuleOutcome) => {
    switch (outcome) {
      case 'BLOCK':
        return (
          <Badge variant="block" icon={<ShieldAlert size={11} />}>
            BLOCK
          </Badge>
        );
      case 'REVIEW':
        return (
          <Badge variant="review" icon={<AlertTriangle size={11} />}>
            REVIEW
          </Badge>
        );
      case 'APPROVE':
        return (
          <Badge variant="approve" icon={<CheckCircle2 size={11} />}>
            APPROVE
          </Badge>
        );
      case 'MONITOR':
        return <Badge variant="neutral">MONITOR</Badge>;
      default:
        return <Badge variant="neutral">{outcome}</Badge>;
    }
  };

  if (rules.length === 0) {
    return (
      <div className="analytics-card empty-rules-card">
        <Sliders size={24} className="text-muted" />
        <p>No deterministic business rules triggered in the selected evaluation window.</p>
      </div>
    );
  }

  return (
    <div className="analytics-card rule-analytics-card">
      <div className="chart-header">
        <div>
          <div className="chart-title-group">
            <Sliders size={16} />
            <h3 className="chart-title">Business Rule Performance &amp; Override Telemetry</h3>
          </div>
          <span className="chart-subtitle">
            Ranked rule trigger frequency, transaction impact, and baseline ML override enforcement across {totalEvaluationsAnalyzed.toLocaleString()} evaluations
          </span>
        </div>

        <div className="rules-count-badge">
          <Badge variant="neutral">{rules.length} Active Rules Triggered</Badge>
        </div>
      </div>

      <div className="rule-table-wrapper">
        <table className="rule-analytics-table">
          <thead>
            <tr>
              <th>Rule Identifier &amp; Type</th>
              <th>Priority</th>
              <th>Intent Description</th>
              <th>Trigger Count</th>
              <th>Affected Txns</th>
              <th>Trigger Rate</th>
              <th>Overrides Enacted</th>
              <th>Outcome Tiers</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((rule) => (
              <tr key={rule.rule_id}>
                <td className="cell-rule-id">
                  <span className="font-mono font-bold text-primary">{rule.rule_id}</span>
                  <span className="rule-type-tag">{rule.rule_type}</span>
                </td>

                <td className="cell-priority font-mono">
                  <span className="priority-badge">#{rule.priority}</span>
                </td>

                <td className="cell-description">
                  <p className="rule-intent">{rule.description}</p>
                </td>

                <td className="cell-triggers font-mono">
                  <strong>{rule.trigger_count.toLocaleString()}</strong>
                </td>

                <td className="cell-affected font-mono">
                  {rule.affected_transactions.toLocaleString()}
                </td>

                <td className="cell-rate font-mono">
                  <div className="rate-wrap">
                    <span>{rule.trigger_rate.toFixed(1)}%</span>
                    <div className="mini-progress-bar">
                      <div
                        className="mini-progress-fill"
                        style={{ width: `${Math.min(100, rule.trigger_rate * 2)}%` }}
                      />
                    </div>
                  </div>
                </td>

                <td className="cell-overrides font-mono">
                  {rule.override_count > 0 ? (
                    <span className="override-badge">
                      <ArrowRightLeft size={11} /> {rule.override_count}
                    </span>
                  ) : (
                    <span className="text-muted">0</span>
                  )}
                </td>

                <td className="cell-outcomes">
                  <div className="outcome-badges-list">
                    {rule.outcomes.map((out) => (
                      <span key={out.outcome} className="outcome-pill" title={`${out.count} (${out.percentage.toFixed(1)}%)`}>
                        {renderOutcomeBadge(out.outcome)}
                        <span className="outcome-count font-mono">{out.count}</span>
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
