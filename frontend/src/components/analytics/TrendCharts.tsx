import React, { useState, useMemo } from 'react';
import {
  TrendingUp,
  Activity,
  ShieldAlert,
  Calendar,
  Layers,
} from 'lucide-react';
import type { TrendDataPoint, TrendInterval } from '../../types/api.ts';

interface TrendChartsProps {
  dataPoints: TrendDataPoint[];
  interval: TrendInterval;
  startDate?: string;
  endDate?: string;
}

export const TrendCharts: React.FC<TrendChartsProps> = ({
  dataPoints,
  interval,
  startDate: _startDate,
  endDate: _endDate,
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Formatter helpers
  const formatTime = (isoString: string): string => {
    try {
      const d = new Date(isoString);
      if (interval === 'hourly') {
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
      }
      return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } catch {
      return isoString;
    }
  };

  const formatFullTime = (isoString: string): string => {
    try {
      const d = new Date(isoString);
      return `${d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    } catch {
      return isoString;
    }
  };

  const formatCurrency = (val: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(val);
  };

  // Aggregates over period
  const totalPeriodTxns = useMemo(() => dataPoints.reduce((acc, p) => acc + p.total_count, 0), [dataPoints]);
  const totalPeriodAmount = useMemo(() => dataPoints.reduce((acc, p) => acc + p.total_amount, 0), [dataPoints]);
  const totalPeriodBlocks = useMemo(() => dataPoints.reduce((acc, p) => acc + p.block_count, 0), [dataPoints]);
  const periodAvgScore = useMemo(() => {
    const scoredPoints = dataPoints.filter((p) => p.total_count > 0);
    if (scoredPoints.length === 0) return 0;
    const sum = scoredPoints.reduce((acc, p) => acc + p.average_risk_score * p.total_count, 0);
    return sum / (totalPeriodTxns || 1);
  }, [dataPoints, totalPeriodTxns]);

  // Max volumes for scaling
  const maxTxnCount = useMemo(() => {
    const max = Math.max(...dataPoints.map((p) => p.total_count), 5);
    return Math.ceil(max * 1.15);
  }, [dataPoints]);

  if (dataPoints.length === 0) {
    return (
      <div className="analytics-card empty-analytics">
        <Activity size={24} className="text-muted" />
        <p>No time-series evaluation telemetry recorded in the selected window.</p>
      </div>
    );
  }

  // SVG dimensions
  const svgWidth = 800;
  const svgHeight = 220;
  const paddingLeft = 50;
  const paddingRight = 30;
  const paddingTop = 25;
  const paddingBottom = 40;
  const plotWidth = svgWidth - paddingLeft - paddingRight;
  const plotHeight = svgHeight - paddingTop - paddingBottom;

  const count = dataPoints.length;
  const getX = (idx: number): number => {
    if (count <= 1) return paddingLeft + plotWidth / 2;
    return paddingLeft + (idx / (count - 1)) * plotWidth;
  };

  // Y for Volume (0 to maxTxnCount)
  const getYVolume = (val: number): number => {
    return paddingTop + plotHeight - (val / (maxTxnCount || 1)) * plotHeight;
  };

  // Y for Score (0 to 100)
  const getYScore = (score: number): number => {
    return paddingTop + plotHeight - (score / 100) * plotHeight;
  };

  // Construct Area Path for Total Volume
  const volumeAreaPath = (() => {
    if (dataPoints.length === 0) return '';
    const points = dataPoints.map((p, idx) => `${getX(idx)},${getYVolume(p.total_count)}`);
    const firstX = getX(0);
    const lastX = getX(dataPoints.length - 1);
    const bottomY = paddingTop + plotHeight;
    return `M ${firstX},${bottomY} L ${points.join(' L ')} L ${lastX},${bottomY} Z`;
  })();

  // Construct Line Path for Block Volume
  const blockLinePath = (() => {
    if (dataPoints.length === 0) return '';
    return 'M ' + dataPoints.map((p, idx) => `${getX(idx)},${getYVolume(p.block_count)}`).join(' L ');
  })();

  // Construct Line Path for Average Risk Score
  const scoreLinePath = (() => {
    if (dataPoints.length === 0) return '';
    return 'M ' + dataPoints.map((p, idx) => `${getX(idx)},${getYScore(p.average_risk_score)}`).join(' L ');
  })();

  const activePoint = hoveredIndex !== null && dataPoints[hoveredIndex] ? dataPoints[hoveredIndex] : null;

  return (
    <div className="analytics-trend-wrapper">
      <div className="trend-kpi-bar">
        <div className="trend-kpi-card">
          <div className="kpi-header">
            <Layers size={14} />
            <span>Period Volume</span>
          </div>
          <div className="kpi-val">{totalPeriodTxns.toLocaleString()}</div>
          <div className="kpi-sub">{formatCurrency(totalPeriodAmount)}</div>
        </div>

        <div className="trend-kpi-card">
          <div className="kpi-header text-block">
            <ShieldAlert size={14} />
            <span>Blocked Threats</span>
          </div>
          <div className="kpi-val text-block">{totalPeriodBlocks.toLocaleString()}</div>
          <div className="kpi-sub font-mono">
            {totalPeriodTxns > 0 ? `${((totalPeriodBlocks / totalPeriodTxns) * 100).toFixed(2)}%` : '0.0%'} Block Rate
          </div>
        </div>

        <div className="trend-kpi-card">
          <div className="kpi-header text-info">
            <TrendingUp size={14} />
            <span>Mean Risk Score</span>
          </div>
          <div className="kpi-val">{periodAvgScore.toFixed(1)}</div>
          <div className="kpi-sub">0–100 Scale Index</div>
        </div>

        <div className="trend-kpi-card">
          <div className="kpi-header">
            <Calendar size={14} />
            <span>Bucket Resolution</span>
          </div>
          <div className="kpi-val uppercase font-mono text-xs">{interval}</div>
          <div className="kpi-sub">{dataPoints.length} Time Buckets</div>
        </div>
      </div>

      {/* Primary Trend Chart: Volume & Blocks */}
      <div className="analytics-card trend-chart-card">
        <div className="chart-header">
          <div className="chart-title-group">
            <h3 className="chart-title">Transaction &amp; Threat Volume Over Time</h3>
            <span className="chart-subtitle">
              Continuous time-series tracking total incoming volume and automated fraud blocks
            </span>
          </div>

          <div className="chart-legend">
            <div className="legend-item">
              <span className="legend-line volume-line" />
              <span>Total Volume</span>
            </div>
            <div className="legend-item">
              <span className="legend-line block-line" />
              <span>Fraud Blocks</span>
            </div>
            <div className="legend-item">
              <span className="legend-line score-line" />
              <span>Mean Risk Score</span>
            </div>
          </div>
        </div>

        <div className="trend-svg-box">
          <svg
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="trend-svg"
            role="img"
            aria-label={`Time series trend chart spanning ${dataPoints.length} ${interval} intervals`}
          >
            <defs>
              <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.35" />
                <stop offset="100%" stopColor="var(--primary)" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Horizontal Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
              const y = paddingTop + plotHeight * (1 - pct);
              const val = Math.round(maxTxnCount * pct);
              return (
                <g key={pct}>
                  <line
                    x1={paddingLeft}
                    y1={y}
                    x2={svgWidth - paddingRight}
                    y2={y}
                    stroke="var(--border-card)"
                    strokeDasharray="2 2"
                    strokeWidth="1"
                  />
                  <text
                    x={paddingLeft - 8}
                    y={y + 4}
                    textAnchor="end"
                    fill="var(--text-muted)"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    {val}
                  </text>
                </g>
              );
            })}

            {/* Volume Area & Line */}
            <path d={volumeAreaPath} fill="url(#volumeGradient)" />
            <path
              d={'M ' + dataPoints.map((p, idx) => `${getX(idx)},${getYVolume(p.total_count)}`).join(' L ')}
              fill="none"
              stroke="var(--primary)"
              strokeWidth="2"
            />

            {/* Block Volume Line */}
            <path
              d={blockLinePath}
              fill="none"
              stroke="var(--status-block)"
              strokeWidth="2.2"
            />

            {/* Mean Score Line (Secondary Axis mapped to height) */}
            <path
              d={scoreLinePath}
              fill="none"
              stroke="var(--accent-cyber)"
              strokeWidth="1.8"
              strokeDasharray="4 2"
            />

            {/* X-axis tick labels */}
            {dataPoints.map((p, idx) => {
              const stepInterval = Math.max(1, Math.floor(count / 7));
              if (idx % stepInterval !== 0 && idx !== count - 1) return null;
              const x = getX(idx);
              return (
                <g key={p.timestamp}>
                  <line
                    x1={x}
                    y1={paddingTop + plotHeight}
                    x2={x}
                    y2={paddingTop + plotHeight + 5}
                    stroke="var(--border-card)"
                    strokeWidth="1"
                  />
                  <text
                    x={x}
                    y={paddingTop + plotHeight + 18}
                    textAnchor="middle"
                    fill="var(--text-muted)"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    {formatTime(p.timestamp)}
                  </text>
                </g>
              );
            })}

            {/* Hover Interaction Vertical Crosshair */}
            {hoveredIndex !== null && (
              <g className="chart-crosshair">
                <line
                  x1={getX(hoveredIndex)}
                  y1={paddingTop}
                  x2={getX(hoveredIndex)}
                  y2={paddingTop + plotHeight}
                  stroke="var(--accent-cyber)"
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                />
                <circle
                  cx={getX(hoveredIndex)}
                  cy={getYVolume(dataPoints[hoveredIndex].total_count)}
                  r="5"
                  fill="var(--primary)"
                  stroke="#fff"
                  strokeWidth="1.5"
                />
                <circle
                  cx={getX(hoveredIndex)}
                  cy={getYVolume(dataPoints[hoveredIndex].block_count)}
                  r="4"
                  fill="var(--status-block)"
                  stroke="#fff"
                  strokeWidth="1.5"
                />
              </g>
            )}

            {/* Invisible interactive column overlays for clean mouse targeting */}
            {dataPoints.map((_, idx) => {
              const colWidth = plotWidth / count;
              const x = getX(idx) - colWidth / 2;
              return (
                <rect
                  key={idx}
                  x={Math.max(paddingLeft, x)}
                  y={paddingTop}
                  width={colWidth}
                  height={plotHeight}
                  fill="transparent"
                  onMouseEnter={() => setHoveredIndex(idx)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  style={{ cursor: 'crosshair' }}
                />
              );
            })}
          </svg>
        </div>

        {/* Hover Point Detailed Breakdown Tooltip */}
        {activePoint && (
          <div className="trend-point-tooltip">
            <div className="tooltip-title font-mono">
              <Calendar size={12} />
              <span>{formatFullTime(activePoint.timestamp)}</span>
            </div>
            <div className="tooltip-metrics">
              <div className="metric-chip">
                <span>Volume:</span>
                <strong>{activePoint.total_count.toLocaleString()} txns</strong>
              </div>
              <div className="metric-chip">
                <span>Amount:</span>
                <strong>{formatCurrency(activePoint.total_amount)}</strong>
              </div>
              <div className="metric-chip text-block">
                <span>Blocks:</span>
                <strong>{activePoint.block_count}</strong>
              </div>
              <div className="metric-chip text-review">
                <span>Reviews:</span>
                <strong>{activePoint.review_count}</strong>
              </div>
              <div className="metric-chip text-approve">
                <span>Approvals:</span>
                <strong>{activePoint.approval_count}</strong>
              </div>
              <div className="metric-chip text-info">
                <span>Mean Risk Score:</span>
                <strong>{activePoint.average_risk_score.toFixed(1)}</strong>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
