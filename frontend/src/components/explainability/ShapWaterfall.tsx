import React, { useState, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Info,
  Maximize2,
  Minimize2,
} from 'lucide-react';
import type { FeatureAttributionDetail } from '../../types/api.ts';

interface ShapWaterfallProps {
  baseValue: number | null;
  outputMargin: number | null;
  modelScore: number;
  featureAttributions: FeatureAttributionDetail[];
}

interface WaterfallStep {
  featureName: string;
  displayName: string;
  rawValue: unknown;
  shapValue: number;
  startValue: number;
  endValue: number;
  direction: 'RISK_INCREASING' | 'MITIGATING';
  relativeContributionPct: number;
  rank: number;
}

export const ShapWaterfall: React.FC<ShapWaterfallProps> = ({
  baseValue,
  outputMargin,
  modelScore,
  featureAttributions,
}) => {
  const [showAll, setShowAll] = useState<boolean>(false);
  const [hoveredStepIndex, setHoveredStepIndex] = useState<number | null>(null);

  // Baseline expected log-odds margin
  const baseVal = baseValue ?? -3.5;
  const finalMargin = outputMargin ?? baseVal;

  // Filter and sort features for waterfall
  const displayedAttributions = useMemo(() => {
    if (!featureAttributions || featureAttributions.length === 0) {
      return [];
    }
    // Sort by absolute SHAP contribution descending
    const sorted = [...featureAttributions].sort(
      (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
    );
    if (!showAll && sorted.length > 8) {
      return sorted.slice(0, 8);
    }
    return sorted;
  }, [featureAttributions, showAll]);

  // Compute sequential waterfall steps in log-odds margin space
  const steps: WaterfallStep[] = useMemo(() => {
    let currentVal = baseVal;
    return displayedAttributions.map((fa) => {
      const start = currentVal;
      const end = currentVal + fa.shap_value;
      currentVal = end;
      return {
        featureName: fa.feature_name,
        displayName: fa.display_name,
        rawValue: fa.raw_value,
        shapValue: fa.shap_value,
        startValue: start,
        endValue: end,
        direction: fa.direction,
        relativeContributionPct: fa.relative_contribution_pct,
        rank: fa.rank,
      };
    });
  }, [baseVal, displayedAttributions]);

  if (featureAttributions.length === 0) {
    return (
      <div className="waterfall-empty">
        <Info size={16} />
        <span>No TreeSHAP attribution records available for local waterfall rendering.</span>
      </div>
    );
  }

  // Calculate SVG scale bounds
  const allValues = [baseVal, finalMargin, ...steps.map((s) => s.startValue), ...steps.map((s) => s.endValue)];
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const valSpan = Math.max(0.001, maxVal - minVal);
  const pad = valSpan * 0.15;
  const scaleMin = minVal - pad;
  const scaleMax = maxVal + pad;
  const totalScaleSpan = scaleMax - scaleMin;

  // SVG Dimension layout
  const rowHeight = 36;
  const marginTop = 40;
  const marginBottom = 45;
  const marginLeft = 180;
  const marginRight = 80;
  const chartWidth = 720;
  const plotWidth = chartWidth - marginLeft - marginRight;
  const totalHeight = marginTop + (steps.length + 2) * rowHeight + marginBottom;

  const valToX = (val: number): number => {
    return marginLeft + ((val - scaleMin) / totalScaleSpan) * plotWidth;
  };

  const baseX = valToX(baseVal);
  const finalX = valToX(finalMargin);

  const formatVal = (val: unknown): string => {
    if (val === null || val === undefined) return '—';
    if (typeof val === 'number') {
      return Number.isInteger(val) ? val.toString() : val.toFixed(2);
    }
    return String(val);
  };

  return (
    <div className="shap-waterfall-card">
      <div className="waterfall-header">
        <div className="waterfall-title-group">
          <h3 className="waterfall-title">TreeSHAP Decision Waterfall</h3>
          <span className="waterfall-subtitle">
            Additive progression in log-odds margin space: <code>E[f(x)] + &Sigma;SHAP = f(x)</code>
          </span>
        </div>

        <div className="waterfall-controls">
          {featureAttributions.length > 8 && (
            <button
              type="button"
              className="waterfall-toggle-btn"
              onClick={() => setShowAll(!showAll)}
              aria-label={showAll ? 'Show top 8 drivers only' : 'Show all local features'}
            >
              {showAll ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              <span>{showAll ? 'Top 8 Key Drivers' : `All ${featureAttributions.length} Drivers`}</span>
            </button>
          )}
        </div>
      </div>

      <div className="waterfall-metrics-banner">
        <div className="waterfall-metric-item">
          <span className="metric-label">Base Expected Margin E[f(x)]</span>
          <span className="metric-val font-mono">{baseVal.toFixed(4)}</span>
        </div>
        <div className="waterfall-metric-item">
          <span className="metric-label">Local Model Margin f(x)</span>
          <span className="metric-val font-mono">
            {finalMargin >= 0 ? `+${finalMargin.toFixed(4)}` : finalMargin.toFixed(4)}
          </span>
        </div>
        <div className="waterfall-metric-item highlight">
          <span className="metric-label">Calibrated Probability P(fraud)</span>
          <span className="metric-val font-mono">{(modelScore * 100).toFixed(2)}%</span>
        </div>
      </div>

      <div className="waterfall-svg-container">
        <svg
          viewBox={`0 0 ${chartWidth} ${totalHeight}`}
          className="waterfall-svg"
          role="img"
          aria-label={`TreeSHAP waterfall visualization explaining risk score with ${steps.length} features`}
        >
          <title>TreeSHAP Decision Waterfall Chart</title>
          <desc>
            Visualizes cumulative log-odds contribution from global baseline {baseVal.toFixed(2)} to final margin {finalMargin.toFixed(2)}.
          </desc>

          {/* Grid lines and ticks */}
          <g className="waterfall-grid">
            {[0.0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
              const tickVal = scaleMin + frac * totalScaleSpan;
              const tickX = valToX(tickVal);
              return (
                <g key={frac}>
                  <line
                    x1={tickX}
                    y1={marginTop - 10}
                    x2={tickX}
                    y2={totalHeight - marginBottom + 5}
                    stroke="var(--border-card)"
                    strokeDasharray="3 3"
                    strokeWidth="1"
                  />
                  <text
                    x={tickX}
                    y={totalHeight - marginBottom + 20}
                    textAnchor="middle"
                    fill="var(--text-muted)"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    {tickVal.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Global Baseline reference guide */}
          <line
            x1={baseX}
            y1={marginTop - 10}
            x2={baseX}
            y2={totalHeight - marginBottom}
            stroke="var(--accent-cyber)"
            strokeWidth="1.5"
            strokeDasharray="4 2"
          />

          {/* Row 0: Baseline Starter Bar */}
          <g className="waterfall-row baseline-row">
            <text
              x={marginLeft - 12}
              y={marginTop + 15}
              textAnchor="end"
              fill="var(--text-secondary)"
              fontSize="11"
              fontWeight="600"
            >
              Baseline E[f(x)]
            </text>
            <circle cx={baseX} cy={marginTop + 12} r="5" fill="var(--accent-cyber)" />
            <text
              x={baseX + (baseVal >= 0 ? 8 : -8)}
              y={marginTop + 16}
              textAnchor={baseVal >= 0 ? 'start' : 'end'}
              fill="var(--accent-cyber)"
              fontSize="11"
              fontFamily="monospace"
              fontWeight="600"
            >
              {baseVal.toFixed(4)}
            </text>
          </g>

          {/* Feature Step Bars */}
          {steps.map((step, idx) => {
            const y = marginTop + (idx + 1) * rowHeight;
            const startX = valToX(step.startValue);
            const endX = valToX(step.endValue);
            const barLeft = Math.min(startX, endX);
            const barWidth = Math.max(3, Math.abs(endX - startX));
            const isPositive = step.shapValue >= 0;
            const barFill = isPositive ? 'var(--status-block)' : 'var(--status-approve)';
            const isHovered = hoveredStepIndex === idx;

            return (
              <g
                key={step.featureName}
                className={`waterfall-step-group ${isHovered ? 'hovered' : ''}`}
                onMouseEnter={() => setHoveredStepIndex(idx)}
                onMouseLeave={() => setHoveredStepIndex(null)}
                tabIndex={0}
                role="graphics-symbol"
                aria-label={`${step.displayName}: ${isPositive ? 'increases risk by' : 'reduces risk by'} ${Math.abs(step.shapValue).toFixed(4)}`}
              >
                {/* Connector line from previous step */}
                <line
                  x1={startX}
                  y1={y - rowHeight + 18}
                  x2={startX}
                  y2={y + 8}
                  stroke="var(--border-card)"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                />

                {/* Feature Name Label */}
                <text
                  x={marginLeft - 12}
                  y={y + 15}
                  textAnchor="end"
                  fill="var(--text-primary)"
                  fontSize="11"
                  fontWeight={isHovered ? '600' : '400'}
                >
                  {step.displayName.length > 22 ? `${step.displayName.slice(0, 20)}...` : step.displayName}
                </text>

                {/* Waterfall Step Bar */}
                <rect
                  x={barLeft}
                  y={y + 4}
                  width={barWidth}
                  height={16}
                  rx="3"
                  fill={barFill}
                  fillOpacity={isHovered ? 1.0 : 0.85}
                  stroke={barFill}
                  strokeWidth={isHovered ? 2 : 1}
                />

                {/* SHAP delta annotation */}
                <text
                  x={isPositive ? barLeft + barWidth + 6 : barLeft - 6}
                  y={y + 16}
                  textAnchor={isPositive ? 'start' : 'end'}
                  fill={isPositive ? 'var(--status-block)' : 'var(--status-approve)'}
                  fontSize="10"
                  fontFamily="monospace"
                  fontWeight="600"
                >
                  {isPositive ? `+${step.shapValue.toFixed(4)}` : step.shapValue.toFixed(4)}
                </text>
              </g>
            );
          })}

          {/* Final Row: Output Model Margin Result */}
          {(() => {
            const finalY = marginTop + (steps.length + 1) * rowHeight;
            return (
              <g className="waterfall-row final-row">
                {/* Connector to final point */}
                {steps.length > 0 && (
                  <line
                    x1={valToX(steps[steps.length - 1].endValue)}
                    y1={finalY - rowHeight + 18}
                    x2={finalX}
                    y2={finalY + 8}
                    stroke="var(--border-card)"
                    strokeWidth="1"
                    strokeDasharray="2 2"
                  />
                )}
                <text
                  x={marginLeft - 12}
                  y={finalY + 15}
                  textAnchor="end"
                  fill="var(--text-primary)"
                  fontSize="11"
                  fontWeight="700"
                >
                  Final Output f(x)
                </text>
                <circle cx={finalX} cy={finalY + 12} r="6" fill="var(--primary)" />
                <text
                  x={finalX + (finalMargin >= 0 ? 10 : -10)}
                  y={finalY + 16}
                  textAnchor={finalMargin >= 0 ? 'start' : 'end'}
                  fill="var(--primary)"
                  fontSize="11"
                  fontFamily="monospace"
                  fontWeight="700"
                >
                  {finalMargin >= 0 ? `+${finalMargin.toFixed(4)}` : finalMargin.toFixed(4)}
                </text>
              </g>
            );
          })()}
        </svg>
      </div>

      {/* Interactive Tooltip Card for Active Step */}
      {hoveredStepIndex !== null && steps[hoveredStepIndex] && (
        <div className="waterfall-tooltip-box">
          <div className="tooltip-header">
            <span className="tooltip-rank font-mono">#{steps[hoveredStepIndex].rank}</span>
            <span className="tooltip-name">{steps[hoveredStepIndex].displayName}</span>
            <span className="tooltip-tech-name font-mono">({steps[hoveredStepIndex].featureName})</span>
          </div>
          <div className="tooltip-grid">
            <div>
              <span className="tooltip-label">Raw Feature Input:</span>
              <span className="tooltip-val font-mono">{formatVal(steps[hoveredStepIndex].rawValue)}</span>
            </div>
            <div>
              <span className="tooltip-label">SHAP Log-Odds Delta:</span>
              <span
                className={`tooltip-val font-mono ${
                  steps[hoveredStepIndex].shapValue >= 0 ? 'text-block' : 'text-approve'
                }`}
              >
                {steps[hoveredStepIndex].shapValue >= 0
                  ? `+${steps[hoveredStepIndex].shapValue.toFixed(4)} (Risk Escalation)`
                  : `${steps[hoveredStepIndex].shapValue.toFixed(4)} (Risk Mitigation)`}
              </span>
            </div>
            <div>
              <span className="tooltip-label">Relative Impact:</span>
              <span className="tooltip-val font-mono">
                {steps[hoveredStepIndex].relativeContributionPct.toFixed(1)}% of directional group
              </span>
            </div>
            <div>
              <span className="tooltip-label">Step Running Margin:</span>
              <span className="tooltip-val font-mono">
                {steps[hoveredStepIndex].startValue.toFixed(4)} &rarr;{' '}
                <strong>{steps[hoveredStepIndex].endValue.toFixed(4)}</strong>
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="waterfall-legend">
        <div className="legend-item">
          <span className="legend-swatch block-swatch" />
          <TrendingUp size={12} className="text-block" />
          <span>Risk-Increasing (&Delta; &gt; 0 log-odds)</span>
        </div>
        <div className="legend-item">
          <span className="legend-swatch approve-swatch" />
          <TrendingDown size={12} className="text-approve" />
          <span>Risk-Mitigating (&Delta; &lt; 0 log-odds)</span>
        </div>
        <div className="legend-item">
          <span className="legend-swatch baseline-swatch" />
          <span>Global Expected Baseline E[f(x)]</span>
        </div>
      </div>
    </div>
  );
};
