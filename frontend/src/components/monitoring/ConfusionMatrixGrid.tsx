import React from 'react';
import type { ConfusionMatrixResponse } from '../../types/monitoring.ts';

interface ConfusionMatrixGridProps {
  matrix: ConfusionMatrixResponse;
  threshold: number;
}

export const ConfusionMatrixGrid: React.FC<ConfusionMatrixGridProps> = ({ matrix, threshold }) => {
  const total = matrix.total > 0 ? matrix.total : 1;
  const tpPct = ((matrix.tp / total) * 100).toFixed(1);
  const fpPct = ((matrix.fp / total) * 100).toFixed(1);
  const fnPct = ((matrix.fn / total) * 100).toFixed(1);
  const tnPct = ((matrix.tn / total) * 100).toFixed(1);

  return (
    <div className="confusion-matrix-wrapper">
      <div className="cm-header">
        <h4 className="cm-title">Classification Confusion Matrix</h4>
        <span className="cm-threshold-tag font-mono">Threshold: {threshold.toFixed(2)}</span>
      </div>

      <div className="confusion-matrix-table">
        {/* Top Header Labels */}
        <div className="cm-corner" />
        <div className="cm-axis-header cm-pred-header">
          <span>Predicted Fraud (Positive)</span>
        </div>
        <div className="cm-axis-header cm-pred-header">
          <span>Predicted Legitimate (Negative)</span>
        </div>

        {/* Row 1: Actual Fraud */}
        <div className="cm-axis-header cm-actual-header">
          <span>Actual Fraud (Positive)</span>
        </div>
        <div className="cm-cell cm-cell-tp" title={`True Positives: ${matrix.tp} cases`}>
          <span className="cm-cell-label">True Positive (TP)</span>
          <strong className="cm-cell-count font-mono">{matrix.tp.toLocaleString()}</strong>
          <span className="cm-cell-pct font-mono">{tpPct}%</span>
        </div>
        <div className="cm-cell cm-cell-fn" title={`False Negatives: ${matrix.fn} cases`}>
          <span className="cm-cell-label">False Negative (FN)</span>
          <strong className="cm-cell-count font-mono">{matrix.fn.toLocaleString()}</strong>
          <span className="cm-cell-pct font-mono">{fnPct}%</span>
        </div>

        {/* Row 2: Actual Legitimate */}
        <div className="cm-axis-header cm-actual-header">
          <span>Actual Legitimate (Negative)</span>
        </div>
        <div className="cm-cell cm-cell-fp" title={`False Positives: ${matrix.fp} cases`}>
          <span className="cm-cell-label">False Positive (FP)</span>
          <strong className="cm-cell-count font-mono">{matrix.fp.toLocaleString()}</strong>
          <span className="cm-cell-pct font-mono">{fpPct}%</span>
        </div>
        <div className="cm-cell cm-cell-tn" title={`True Negatives: ${matrix.tn} cases`}>
          <span className="cm-cell-label">True Negative (TN)</span>
          <strong className="cm-cell-count font-mono">{matrix.tn.toLocaleString()}</strong>
          <span className="cm-cell-pct font-mono">{tnPct}%</span>
        </div>
      </div>

      <div className="cm-summary-footer">
        <span>Total Ground-Truth Samples: <strong className="font-mono">{matrix.total.toLocaleString()}</strong></span>
      </div>
    </div>
  );
};
