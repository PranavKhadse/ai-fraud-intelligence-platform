import React, { useState, useEffect, useCallback } from 'react';
import {
  Play,
  RotateCcw,
  Sparkles,
  ShieldAlert,
  Sliders,
  FileText,
  Clock,
  Zap,
  XCircle,
} from 'lucide-react';
import { FeatureCategoryEditor } from './FeatureCategoryEditor.tsx';
import { SimulationComparisonView } from './SimulationComparisonView.tsx';
import { ShapWaterfall } from '../explainability/ShapWaterfall.tsx';
import { RiskScoreMeter } from '../common/RiskScoreMeter.tsx';
import { Badge } from '../common/Badge.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { simulateTransactionScenario } from '../../api/dashboardApi.ts';
import {
  FEATURE_CATEGORIES,
  getDefaultSimulatedFeatures,
  DEMO_PRESETS,
  type DemoPreset,
} from '../../constants/featureCatalog.ts';
import type {
  SimulationRequest,
  SimulationResponse,
  TransactionDetailResponse,
} from '../../types/api.ts';

interface WhatIfSimulatorProps {
  initialBaselineTransaction?: TransactionDetailResponse | null;
  onClearBaseline?: () => void;
}

export const WhatIfSimulator: React.FC<WhatIfSimulatorProps> = ({
  initialBaselineTransaction,
  onClearBaseline,
}) => {
  const [baselineTx, setBaselineTx] = useState<TransactionDetailResponse | null>(
    initialBaselineTransaction || null
  );

  const [features, setFeatures] = useState<Record<string, any>>(() => {
    if (initialBaselineTransaction?.features?.features_snapshot) {
      return { ...initialBaselineTransaction.features.features_snapshot };
    }
    return getDefaultSimulatedFeatures();
  });

  const [baselineFeatures, setBaselineFeatures] = useState<Record<string, any>>(() => {
    if (initialBaselineTransaction?.features?.features_snapshot) {
      return { ...initialBaselineTransaction.features.features_snapshot };
    }
    return {};
  });

  const [simulationResult, setSimulationResult] = useState<SimulationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Sync state if initialBaselineTransaction prop updates
  useEffect(() => {
    if (initialBaselineTransaction) {
      setBaselineTx(initialBaselineTransaction);
      if (initialBaselineTransaction.features?.features_snapshot) {
        const snap = { ...initialBaselineTransaction.features.features_snapshot };
        setFeatures(snap);
        setBaselineFeatures(snap);
      }
    }
  }, [initialBaselineTransaction]);

  const handleFeatureChange = useCallback((name: string, value: any) => {
    setFeatures((prev) => ({
      ...prev,
      [name]: value,
    }));
  }, []);

  const handleResetSingleFeature = useCallback(
    (name: string) => {
      if (baselineFeatures[name] !== undefined) {
        setFeatures((prev) => ({
          ...prev,
          [name]: baselineFeatures[name],
        }));
      } else {
        const defaults = getDefaultSimulatedFeatures();
        setFeatures((prev) => ({
          ...prev,
          [name]: defaults[name],
        }));
      }
    },
    [baselineFeatures]
  );

  const handleResetAllToBaseline = useCallback(() => {
    if (Object.keys(baselineFeatures).length > 0) {
      setFeatures({ ...baselineFeatures });
    } else {
      setFeatures(getDefaultSimulatedFeatures());
    }
  }, [baselineFeatures]);

  const handleApplyPreset = useCallback((preset: DemoPreset) => {
    setFeatures((prev) => ({
      ...prev,
      ...preset.overrides,
    }));
  }, []);

  const handleClearBaseline = useCallback(() => {
    setBaselineTx(null);
    setBaselineFeatures({});
    if (onClearBaseline) onClearBaseline();
  }, [onClearBaseline]);

  const runSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);

    const payload: SimulationRequest = {
      simulated_features: features,
      baseline_transaction_id: baselineTx?.transaction?.id,
      top_k: 5,
      top_mitigating: 3,
      max_reasons: 5,
    };

    try {
      const response = await simulateTransactionScenario(payload);
      setSimulationResult(response);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Simulation evaluation failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [features, baselineTx]);

  // Keyboard shortcut: Ctrl + Enter to run simulation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runSimulation();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [runSimulation]);

  return (
    <div className="what-if-simulator-container">
      {/* 1. Watermark & Read-Only Banner */}
      <div className="simulator-watermark-banner" role="status">
        <div className="watermark-content">
          <ShieldAlert size={16} />
          <span>
            SIMULATION SCENARIO — READ-ONLY / NOT PERSISTED (No database mutation or audit trail generated)
          </span>
        </div>
      </div>

      {/* 2. Simulator Header Controls */}
      <div className="simulator-header-card">
        <div className="simulator-header-top">
          <div>
            <h1 className="simulator-title">What-If Transaction Risk Simulator</h1>
            <p className="simulator-subtitle">
              Modify candidate 55-feature inputs to simulate real-time ML inference, rule actions, and TreeSHAP explainability shifts.
            </p>
          </div>

          <div className="simulator-actions-bar">
            <button
              type="button"
              className="btn-secondary"
              onClick={handleResetAllToBaseline}
              title="Reset all fields to baseline"
            >
              <RotateCcw size={14} />
              <span>Reset to Baseline</span>
            </button>
            <button
              type="button"
              className="btn-primary run-simulation-btn"
              onClick={runSimulation}
              disabled={loading}
              title="Run simulation (Ctrl + Enter)"
            >
              <Play size={15} />
              <span>{loading ? 'Simulating...' : 'Run Simulation'}</span>
            </button>
          </div>
        </div>

        {/* Preset Scenarios Chips */}
        <div className="preset-scenarios-bar">
          <div className="preset-label">
            <Sparkles size={14} />
            <span>Example Demo Presets:</span>
          </div>
          <div className="preset-chips">
            {DEMO_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className="preset-chip"
                onClick={() => handleApplyPreset(preset)}
                title={preset.description}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        {/* Baseline Status Pill */}
        {baselineTx && (
          <div className="baseline-indicator-banner">
            <div className="baseline-info-text">
              <span className="baseline-badge">Active Baseline</span>
              <span>Transaction ID: <strong>{baselineTx.transaction.external_transaction_id}</strong> ({baselineTx.transaction.id})</span>
              <span>• Original Risk: <strong>{baselineTx.evaluation?.risk_score}</strong> ({baselineTx.evaluation?.decision_action})</span>
            </div>
            <button
              type="button"
              className="clear-baseline-btn"
              onClick={handleClearBaseline}
              title="Clear baseline comparison"
            >
              <XCircle size={14} />
              <span>Detach Baseline</span>
            </button>
          </div>
        )}
      </div>

      {error && (
        <ErrorBanner
          title="Simulation Evaluation Error"
          message={error}
          onRetry={runSimulation}
        />
      )}

      {/* 3. Main Workspace Grid: Features Editor (Left) & Results View (Right) */}
      <div className="simulator-workspace-layout">
        {/* Left Column: 7-Category Feature Accordions */}
        <div className="simulator-editor-column">
          <div className="editor-column-header">
            <h2 className="column-title">
              <Sliders size={18} />
              <span>Candidate Feature Vector (55 Features)</span>
            </h2>
            <span className="editor-hint">Changes automatically highlight</span>
          </div>

          <div className="feature-categories-stack">
            {FEATURE_CATEGORIES.map((cat, idx) => (
              <FeatureCategoryEditor
                key={cat}
                category={cat}
                features={features}
                baselineFeatures={baselineFeatures}
                onChange={handleFeatureChange}
                onResetFeature={handleResetSingleFeature}
                defaultExpanded={idx === 0 || idx === 1}
              />
            ))}
          </div>
        </div>

        {/* Right Column: Simulation Results & Waterfall */}
        <div className="simulator-results-column">
          <div className="results-column-header">
            <h2 className="column-title">
              <Zap size={18} />
              <span>Simulation Outcome &amp; Impact</span>
            </h2>
            {simulationResult && (
              <span className="latency-badge">
                <Clock size={12} />
                <span>{simulationResult.evaluation_latency_ms.toFixed(1)} ms</span>
              </span>
            )}
          </div>

          {loading ? (
            <div className="results-loading-card">
              <LoadingSpinner message="Evaluating ML decision policy and TreeSHAP explainability in-memory..." />
            </div>
          ) : simulationResult ? (
            <div className="simulation-results-content">
              {/* Baseline Comparison Card (if baseline exists) */}
              {simulationResult.baseline && simulationResult.comparison && (
                <SimulationComparisonView
                  baseline={simulationResult.baseline}
                  simulated={simulationResult.simulated}
                  comparison={simulationResult.comparison}
                />
              )}

              {/* Simulated Outcome KPI Bar */}
              <div className="simulated-outcome-card">
                <div className="outcome-meter-row">
                  <div className="meter-wrapper">
                    <RiskScoreMeter
                      score={simulationResult.simulated.risk_score}
                      tier={simulationResult.simulated.risk_tier}
                    />
                  </div>
                  <div className="outcome-details">
                    <div className="outcome-action-row">
                      <span className="outcome-label">Simulated Decision:</span>
                      <Badge variant={simulationResult.simulated.decision_action.toLowerCase() as any}>
                        {simulationResult.simulated.decision_action}
                      </Badge>
                      <Badge variant={simulationResult.simulated.risk_tier.toLowerCase() as any}>
                        {simulationResult.simulated.risk_tier}
                      </Badge>
                      {simulationResult.simulated.is_overridden && (
                        <Badge variant="block">RULE OVERRIDE</Badge>
                      )}
                    </div>
                    <p className="outcome-reason-text">
                      {simulationResult.simulated.decision_reason}
                    </p>
                    <div className="outcome-prob-stat">
                      Continuous Model Probability: <strong>{(simulationResult.simulated.model_score * 100).toFixed(2)}%</strong>
                      {simulationResult.simulated.output_margin !== null && simulationResult.simulated.output_margin !== undefined && (
                        <span> | Log-Odds Margin: <code>{simulationResult.simulated.output_margin.toFixed(4)}</code></span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Simulated TreeSHAP Local Explainability Waterfall */}
              {simulationResult.simulated.feature_attributions.length > 0 && (
                <div className="results-waterfall-card">
                  <ShapWaterfall
                    baseValue={simulationResult.simulated.base_value ?? 0.0}
                    outputMargin={simulationResult.simulated.output_margin ?? 0.0}
                    modelScore={simulationResult.simulated.model_score}
                    featureAttributions={simulationResult.simulated.feature_attributions}
                  />
                </div>
              )}

              {/* Simulated Reason Codes List */}
              {simulationResult.simulated.reason_codes.length > 0 && (
                <div className="results-reasons-card">
                  <h3 className="section-title">
                    <FileText size={16} />
                    <span>Synthesized Reason Codes ({simulationResult.simulated.reason_codes.length})</span>
                  </h3>
                  <div className="reasons-list">
                    {simulationResult.simulated.reason_codes.map((rc) => (
                      <div key={rc.id} className="reason-item-card">
                        <div className="reason-item-header">
                          <span className="reason-code-badge mono-text">{rc.code}</span>
                          <span className="reason-headline">{rc.headline}</span>
                          <Badge variant={rc.severity.toLowerCase() as any}>{rc.severity}</Badge>
                        </div>
                        <p className="reason-desc">{rc.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Simulated Triggered Rules List */}
              {simulationResult.simulated.rule_matches.length > 0 && (
                <div className="results-rules-card">
                  <h3 className="section-title">
                    <ShieldAlert size={16} />
                    <span>Triggered Business Rules ({simulationResult.simulated.rule_matches.length})</span>
                  </h3>
                  <div className="rules-list">
                    {simulationResult.simulated.rule_matches.map((rm) => (
                      <div key={rm.id} className="rule-item-card">
                        <div className="rule-item-header">
                          <span className="mono-text font-bold">{rm.rule_id}</span>
                          <span className="rule-desc">{rm.description}</span>
                          <Badge variant={rm.outcome.toLowerCase() as any}>{rm.outcome}</Badge>
                        </div>
                        <div className="rule-item-sub">
                          <span>Feature: <code>{rm.feature_name}</code></span>
                          <span>Operator: <code>{rm.operator} {rm.comparison_value}</code></span>
                          <span>Priority: {rm.priority}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-simulation-card">
              <Sliders size={36} />
              <h3>Ready to Simulate</h3>
              <p>
                Adjust features on the left or click a demo preset above, then click <strong>Run Simulation</strong> (or press <code>Ctrl + Enter</code>) to evaluate risk outcomes in-memory.
              </p>
              <button
                type="button"
                className="btn-primary"
                onClick={runSimulation}
              >
                <Play size={14} />
                <span>Execute Scenario Simulation</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
