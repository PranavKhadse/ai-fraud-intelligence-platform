import React, { useState } from 'react';
import { X, CheckCircle2, AlertCircle, ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';
import { submitCaseDisposition } from '../../api/caseApi.ts';
import { ApiClientError } from '../../api/client.ts';
import type { CaseDisposition, CaseResponse } from '../../types/case.ts';

interface DispositionModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseId: string;
  caseNumber: string;
  onSuccess: (updatedCase: CaseResponse) => void;
}

const DISPOSITION_OPTIONS: {
  value: CaseDisposition;
  label: string;
  description: string;
  tone: 'block' | 'approve' | 'review' | 'neutral';
}[] = [
  {
    value: 'CONFIRMED_FRAUD',
    label: 'Confirmed Fraud',
    description: 'Transaction confirmed as unauthorized, synthetic identity, or account takeover.',
    tone: 'block',
  },
  {
    value: 'FALSE_POSITIVE',
    label: 'False Positive',
    description: 'Legitimate cardholder activity erroneously flagged or elevated by risk rules.',
    tone: 'approve',
  },
  {
    value: 'LEGITIMATE',
    label: 'Legitimate Customer Transaction',
    description: 'Verified genuine transaction following manual analyst review.',
    tone: 'approve',
  },
  {
    value: 'SUSPICIOUS_RESOLVED',
    label: 'Suspicious Activity Resolved',
    description: 'Borderline or elevated anomaly investigated and deemed acceptable / mitigated.',
    tone: 'review',
  },
];

export const DispositionModal: React.FC<DispositionModalProps> = ({
  isOpen,
  onClose,
  caseId,
  caseNumber,
  onSuccess,
}) => {
  const [disposition, setDisposition] = useState<CaseDisposition>('CONFIRMED_FRAUD');
  const [reason, setReason] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reason.trim() || reason.trim().length < 10) {
      setError('Disposition rationale must be at least 10 characters long.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const updated = await submitCaseDisposition(caseId, {
        disposition,
        reason: reason.trim(),
      });
      onSuccess(updated);
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.statusCode === 403) {
          setError('Permission Denied: Only analysts or administrators can submit case dispositions.');
        } else if (err.statusCode === 409) {
          setError(err.message || 'Conflict: Case status cannot be dispositioned from its current state.');
        } else if (err.statusCode === 422) {
          setError(err.message || 'Validation Error: Disposition rationale must be at least 10 characters.');
        } else {
          setError(err.message || 'Failed to submit disposition.');
        }
      } else {
        const msg = err instanceof Error ? err.message : 'Unknown error submitting disposition';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="disposition-title">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <CheckCircle2 size={20} className="modal-icon-success" />
            <h3 id="disposition-title" className="modal-title">
              Submit Human Review Disposition
            </h3>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close modal">
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="modal-body">
            <div className="form-field-info">
              <span className="field-info-label">Case Number:</span>
              <span className="field-info-val font-mono highlight">{caseNumber}</span>
            </div>

            {error && (
              <div className="form-error-banner" role="alert">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            <div className="form-group">
              <label className="form-label">
                Final Investigation Outcome <span className="required-star">*</span>
              </label>
              <div className="disposition-options-grid">
                {DISPOSITION_OPTIONS.map((opt) => (
                  <label
                    key={opt.value}
                    className={`disposition-option-card ${
                      disposition === opt.value ? `selected selected-${opt.tone}` : ''
                    }`}
                  >
                    <input
                      type="radio"
                      name="disposition-choice"
                      value={opt.value}
                      checked={disposition === opt.value}
                      onChange={() => setDisposition(opt.value)}
                      disabled={loading}
                      className="sr-only"
                    />
                    <div className="option-header">
                      {opt.tone === 'block' ? (
                        <ShieldAlert size={16} className="text-block" />
                      ) : opt.tone === 'approve' ? (
                        <ShieldCheck size={16} className="text-approve" />
                      ) : (
                        <AlertTriangle size={16} className="text-review" />
                      )}
                      <strong>{opt.label}</strong>
                    </div>
                    <p className="option-desc">{opt.description}</p>
                  </label>
                ))}
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="disposition-reason" className="form-label">
                Authoritative Review Justification &amp; Rationale <span className="required-star">*</span>
              </label>
              <textarea
                id="disposition-reason"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Explain the definitive investigation findings, customer contact outcomes, or documentary evidence (minimum 10 characters)..."
                className="form-textarea"
                rows={4}
                disabled={loading}
                required
              />
              <div className="field-footer">
                <span className="field-hint">
                  This disposition will mark the case as RESOLVED and freeze further risk edits.
                </span>
                <span className={`char-count ${reason.trim().length < 10 ? 'text-warning' : 'text-success'}`}>
                  {reason.trim().length}/10 min chars
                </span>
              </div>
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-modal-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn-modal-primary"
              disabled={loading || reason.trim().length < 10}
            >
              {loading ? 'Recording Disposition...' : 'Resolve Case with Disposition'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
