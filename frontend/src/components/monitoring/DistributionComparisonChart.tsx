import React, { useState } from 'react';

interface DistributionComparisonChartProps {
  title?: string;
  subtitle?: string;
  baselineDistribution: Record<string, number>;
  currentDistribution: Record<string, number>;
  binEdges?: number[] | null;
  height?: number;
}

export const DistributionComparisonChart: React.FC<DistributionComparisonChartProps> = ({
  title,
  subtitle,
  baselineDistribution,
  currentDistribution,
  binEdges,
  height = 180,
}) => {
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);

  // Collect all unique keys in order
  const allKeys = Array.from(
    new Set([...Object.keys(baselineDistribution), ...Object.keys(currentDistribution)])
  );

  // Natural sort if keys are bin_0, bin_1, or bucket_0, bucket_1
  allKeys.sort((a, b) => {
    const numA = parseInt(a.replace(/\D/g, ''), 10);
    const numB = parseInt(b.replace(/\D/g, ''), 10);
    if (!isNaN(numA) && !isNaN(numB)) {
      return numA - numB;
    }
    return a.localeCompare(b);
  });

  if (allKeys.length === 0) {
    return (
      <div className="dist-chart-empty">
        <span>No distribution data available for comparison.</span>
      </div>
    );
  }

  // Find maximum proportion for scaling
  const maxVal = Math.max(
    ...allKeys.map((k) => Math.max(baselineDistribution[k] || 0, currentDistribution[k] || 0)),
    0.05
  );

  const formatLabel = (key: string, idx: number): string => {
    if (binEdges && binEdges.length > idx + 1) {
      const low = binEdges[idx];
      const high = binEdges[idx + 1];
      const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(2));
      return `[${fmt(low)}, ${fmt(high)})`;
    }
    return key.replace(/_/g, ' ');
  };

  return (
    <div className="distribution-comparison-container">
      {title && (
        <div className="dist-chart-header">
          <div>
            <h4 className="dist-chart-title">{title}</h4>
            {subtitle && <span className="dist-chart-subtitle">{subtitle}</span>}
          </div>
          <div className="dist-chart-legend">
            <div className="legend-item">
              <span className="legend-swatch swatch-baseline" />
              <span>Baseline (Frozen)</span>
            </div>
            <div className="legend-item">
              <span className="legend-swatch swatch-current" />
              <span>Current Window</span>
            </div>
          </div>
        </div>
      )}

      <div className="dist-bars-wrapper" style={{ minHeight: `${height}px` }}>
        {allKeys.map((key, idx) => {
          const basePct = baselineDistribution[key] || 0;
          const currPct = currentDistribution[key] || 0;
          const baseHeight = Math.max(2, (basePct / maxVal) * 100);
          const currHeight = Math.max(2, (currPct / maxVal) * 100);
          const isHovered = hoveredKey === key;
          const label = formatLabel(key, idx);

          return (
            <div
              key={key}
              className={`dist-bar-pair ${isHovered ? 'hovered' : ''}`}
              onMouseEnter={() => setHoveredKey(key)}
              onMouseLeave={() => setHoveredKey(null)}
              tabIndex={0}
              role="graphics-symbol"
              aria-label={`${label}: Baseline ${(basePct * 100).toFixed(1)}%, Current ${(currPct * 100).toFixed(1)}%`}
            >
              <div className="bars-track">
                {/* Baseline bar */}
                <div
                  className="bar bar-baseline"
                  style={{ height: `${baseHeight}%` }}
                  title={`Baseline: ${(basePct * 100).toFixed(2)}%`}
                />
                {/* Current bar */}
                <div
                  className="bar bar-current"
                  style={{ height: `${currHeight}%` }}
                  title={`Current: ${(currPct * 100).toFixed(2)}%`}
                />
              </div>

              <div className="bar-axis-label font-mono" title={label}>
                {label}
              </div>
            </div>
          );
        })}
      </div>

      {hoveredKey !== null && (
        <div className="dist-hover-detail-card">
          <div className="detail-col">
            <span className="lbl">Bin / Category:</span>
            <strong className="val font-mono">
              {formatLabel(hoveredKey, allKeys.indexOf(hoveredKey))}
            </strong>
          </div>
          <div className="detail-col">
            <span className="lbl">Baseline Proportion:</span>
            <strong className="val font-mono text-cyan">
              {((baselineDistribution[hoveredKey] || 0) * 100).toFixed(2)}%
            </strong>
          </div>
          <div className="detail-col">
            <span className="lbl">Current Proportion:</span>
            <strong className="val font-mono text-primary">
              {((currentDistribution[hoveredKey] || 0) * 100).toFixed(2)}%
            </strong>
          </div>
          <div className="detail-col">
            <span className="lbl">Delta:</span>
            <strong
              className={`val font-mono ${
                Math.abs((currentDistribution[hoveredKey] || 0) - (baselineDistribution[hoveredKey] || 0)) > 0.05
                  ? 'text-amber'
                  : ''
              }`}
            >
              {(
                ((currentDistribution[hoveredKey] || 0) - (baselineDistribution[hoveredKey] || 0)) *
                100
              ).toFixed(2)}
              %
            </strong>
          </div>
        </div>
      )}
    </div>
  );
};
