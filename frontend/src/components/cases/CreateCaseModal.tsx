import React, { useState } from 'react';
import { X, ShieldAlert, AlertCircle, ArrowRight } from 'lucide-react';
import { createManualCase } from '../../api/caseApi.ts';

import { ApiClientError } from '../../api/client.ts';
import type { CasePriority, CaseResponse } from '../../types/case.ts';

interface CreateCaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  transactionId: string;
  evaluationId?: string | null;
  onSuccess: (createdCase: CaseResponse) => void;
  onOpenExistingCase?: (caseId: string) => void;
}

export const CreateCaseModal: React.FC<CreateCaseModalProps> = ({
  isOpen,
  onClose,
  transactionId,
  evaluationId,
  onSuccess,
  onOpenExistingCase,
}) => {
  const [priority, setPriority] = useState<CasePriority>('HIGH');
  const [initialNote, setInitialNote] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [conflictCase, setConflictCase] = useState<{ id?: string; caseNumber?: string } | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!initialNote.trim() || initialNote.trim().length < 5) {
      setError('Initial investigation note must contain at least 5 characters.');
      return;
    }

    setLoading(true);
    setError(null);
    setConflictCase(null);

    try {
      const created = await createManualCase({
        transaction_id: transactionId,
        evaluation_id: evaluationId || null,
        priority,
        initial_note: initialNote.trim(),
      });
      onSuccess(created);
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.statusCode === 409) {
          // Check for existing case details
          const data = err.data as { detail?: string; details?: { existing_case_id?: string; existing_case_number?: string } } | undefined;
          const existingId = data?.details?.existing_case_id;
          const existingNumber = data?.details?.existing_case_number;

          if (existingId) {
            setConflictCase({ id: existingId, caseNumber: existingNumber });
          } else {
            setError(err.message || 'A case already exists for this transaction.');
          }
        } else if (err.statusCode === 403) {
          setError('Permission Denied: Only analysts or administrators can manually escalate cases.');
        } else if (err.statusCode === 422) {
          setError(err.message || 'Validation Error: Check input fields and retry.');
        } else {
          setError(err.message || 'Failed to create manual case.');
        }
      } else {
        const msg = err instanceof Error ? err.message : 'Unknown error creating case';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="create-case-title">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <ShieldAlert size={20} className="modal-icon-alert" />
            <h3 id="create-case-title" className="modal-title">
              Manual Case Escalation
            </h3>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close modal">
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="modal-body">
            <div className="form-field-info">
              <span className="field-info-label">Target Transaction ID:</span>
              <span className="field-info-val font-mono">{transactionId}</span>
            </div>

            {conflictCase && (
              <div className="conflict-banner" role="alert">
                <div className="conflict-header">
                  <AlertCircle size={18} className="text-warning" />
                  <strong>Case Already Exists for this Transaction</strong>
                </div>
                <p className="conflict-message">
                  {conflictCase.caseNumber
                    ? `Case ${conflictCase.caseNumber} has already been opened for transaction ${transactionId.slice(0, 10)}...`
                    : 'A case is already registered for this transaction in PostgreSQL.'}
                </p>
                {conflictCase.id && onOpenExistingCase && (
                  <button
                    type="button"
                    className="btn-open-existing"
                    onClick={() => {
                      onOpenExistingCase(conflictCase.id!);
                      onClose();
                    }}
                  >
                    <span>Open Existing Case Workspace</span>
                    <ArrowRight size={14} />
                  </button>
                )}
              </div>
            )}

            {error && !conflictCase && (
              <div className="form-error-banner" role="alert">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            {!conflictCase && (
              <>
                <div className="form-group">
                  <label htmlFor="case-priority-select" className="form-label">
                    Triage Priority <span className="required-star">*</span>
                  </label>
                  <select
                    id="case-priority-select"
                    value={priority}
                    onChange={(e) => setPriority(e.target.value as CasePriority)}
                    className="form-select"
                    disabled={loading}
                  >
                    <option value="CRITICAL">CRITICAL — High Financial Exposure / Imminent Loss</option>
                    <option value="HIGH">HIGH — Suspicious Velocity / High Risk Score</option>
                    <option value="MEDIUM">MEDIUM — Moderate Anomaly / Standard Triage</option>
                    <option value="LOW">LOW — Routine Review / Low Risk Score</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="case-initial-note" className="form-label">
                    Initial Investigation Note <span className="required-star">*</span>
                  </label>
                  <textarea
                    id="case-initial-note"
                    value={initialNote}
                    onChange={(e) => setInitialNote(e.target.value)}
                    placeholder="Enter analyst escalation reason, observed behavioral anomalies, or suspicious patterns (minimum 5 characters)..."
                    className="form-textarea"
                    rows={4}
                    disabled={loading}
                    required
                  />
                  <span className="field-hint">
                    This note will be recorded as the opening audit entry in the case timeline.
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-modal-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            {!conflictCase && (
              <button
                type="submit"
                className="btn-modal-primary"
                disabled={loading || initialNote.trim().length < 5}
              >
                {loading ? 'Creating Case...' : 'Escalate to Review Queue'}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};
