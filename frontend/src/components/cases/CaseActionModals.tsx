import React, { useState } from 'react';
import { X, UserPlus, AlertTriangle, CheckCircle, RotateCcw, AlertCircle } from 'lucide-react';
import { updateCaseAssignment, updateCaseStatus } from '../../api/caseApi.ts';

import { ApiClientError } from '../../api/client.ts';
import type { CaseResponse } from '../../types/case.ts';

interface BaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseId: string;
  caseNumber: string;
  onSuccess: (updatedCase: CaseResponse) => void;
}

// =============================================================================
// 1. Assign Modal (Admin / Lead Analyst Reassignment)
// =============================================================================

interface AssignModalProps extends BaseModalProps {
  currentAssignee: string | null;
}

export const AssignModal: React.FC<AssignModalProps> = ({
  isOpen,
  onClose,
  caseId,
  caseNumber,
  currentAssignee,
  onSuccess,
}) => {
  const [assigneeId, setAssigneeId] = useState<string>('');
  const [reason, setReason] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!assigneeId.trim()) {
      setError('Please provide a target analyst identifier.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const updated = await updateCaseAssignment(caseId, {
        action: 'ASSIGN',
        assignee_id: assigneeId.trim(),
        reason: reason.trim() || undefined,
      });
      onSuccess(updated);
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.statusCode === 403) {
          setError('Permission Denied: Only administrators can assign cases to other analysts.');
        } else if (err.statusCode === 409) {
          setError(err.message || 'Conflict: Assignment could not be modified in the current case state.');
        } else {
          setError(err.message || 'Failed to update assignment.');
        }
      } else {
        const msg = err instanceof Error ? err.message : 'Unknown error updating assignment';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="assign-title">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <UserPlus size={20} className="modal-icon-info" />
            <h3 id="assign-title" className="modal-title">
              Assign Case to Analyst
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
            {currentAssignee && (
              <div className="form-field-info">
                <span className="field-info-label">Currently Assigned To:</span>
                <span className="field-info-val font-mono">{currentAssignee}</span>
              </div>
            )}

            {error && (
              <div className="form-error-banner" role="alert">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            <div className="form-group">
              <label htmlFor="assignee-id-input" className="form-label">
                Target Analyst Identifier <span className="required-star">*</span>
              </label>
              <input
                id="assignee-id-input"
                type="text"
                value={assigneeId}
                onChange={(e) => setAssigneeId(e.target.value)}
                placeholder="e.g. analyst_02, analyst_senior"
                className="form-input"
                disabled={loading}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="assign-reason-input" className="form-label">
                Assignment Reason / Instructions (Optional)
              </label>
              <input
                id="assign-reason-input"
                type="text"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="e.g. Reassigned for deep merchant category investigation"
                className="form-input"
                disabled={loading}
              />
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-modal-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn-modal-primary" disabled={loading || !assigneeId.trim()}>
              {loading ? 'Assigning...' : 'Assign Case'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// =============================================================================
// 2. Escalate Modal (Escalate to Senior Triage / Lead Tier)
// =============================================================================

export const EscalateModal: React.FC<BaseModalProps> = ({
  isOpen,
  onClose,
  caseId,
  caseNumber,
  onSuccess,
}) => {
  const [reason, setReason] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reason.trim() || reason.trim().length < 5) {
      setError('Escalation justification must be at least 5 characters long.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const updated = await updateCaseStatus(caseId, {
        target_status: 'ESCALATED',
        reason: reason.trim(),
      });
      onSuccess(updated);
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.statusCode === 403) {
          setError('Permission Denied: You are not authorized to escalate this case.');
        } else if (err.statusCode === 409) {
          setError(err.message || 'Conflict: Only OPEN or IN_REVIEW cases can be escalated.');
        } else if (err.statusCode === 422) {
          setError(err.message || 'Validation Error: Escalation reason must be at least 5 characters.');
        } else {
          setError(err.message || 'Failed to escalate case.');
        }
      } else {
        const msg = err instanceof Error ? err.message : 'Unknown error escalating case';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="escalate-title">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <AlertTriangle size={20} className="modal-icon-alert" />
            <h3 id="escalate-title" className="modal-title">
              Escalate Case for Senior Triage
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
              <label htmlFor="escalate-reason" className="form-label">
                Escalation Justification <span className="required-star">*</span>
              </label>
              <textarea
                id="escalate-reason"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="State why this case requires senior analyst or fraud management escalation (minimum 5 characters)..."
                className="form-textarea"
                rows={4}
                disabled={loading}
                required
              />
              <span className="field-hint">
                This will transition the case to ESCALATED status and append an ESCALATION note.
              </span>
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-modal-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn-modal-warning" disabled={loading || reason.trim().length < 5}>
              {loading ? 'Escalating...' : 'Confirm Escalation'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// =============================================================================
// 3. Close Modal (Administrative or Resolved Case Closure)
// =============================================================================

export const CloseModal: React.FC<BaseModalProps> = ({
  isOpen,
  onClose,
  caseId,
  caseNumber,
  onSuccess,
}) => {
  const [reason, setReason] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reason.trim() || reason.trim().length < 5) {
      setError('Closure reason must be at least 5 characters long.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const updated = await updateCaseStatus(caseId, {
        target_status: 'CLOSED',
        reason: reason.trim(),
      });
      onSuccess(updated);
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.statusCode === 403) {
          setError('Permission Denied: You are not authorized to close this case.');
        } else if (err.statusCode === 409) {
          setError(err.message || 'Conflict: Case status cannot transition to CLOSED from current state.');
        } else if (err.statusCode === 422) {
          setError(err.message || 'Validation Error: Closure reason must be at least 5 characters.');
        } else {
          setError(err.message || 'Failed to close case.');
        }
      } else {
        const msg = err instanceof Error ? err.message : 'Unknown error closing case';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="close-case-title">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <CheckCircle size={20} className="modal-icon-neutral" />
            <h3 id="close-case-title" className="modal-title">
              Close Case Investigation
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
              <label htmlFor="close-reason" className="form-label">
                Closure Rationale <span className="required-star">*</span>
              </label>
              <textarea
                id="close-reason"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Explain the reason for closing this investigation without/after disposition (minimum 5 characters)..."
                className="form-textarea"
                rows={4}
                disabled={loading}
                required
              />
              <span className="field-hint">
                Transitions case to CLOSED status and logs the closing timestamp.
              </span>
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-modal-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn-modal-primary" disabled={loading || reason.trim().length < 5}>
              {loading ? 'Closing...' : 'Confirm Close Case'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// =============================================================================
// 4. Reopen Modal (Reopening a Resolved or Closed Case)
// =============================================================================

export const ReopenModal: React.FC<BaseModalProps> = ({
  isOpen,
  onClose,
  caseId,
  caseNumber,
  onSuccess,
}) => {
  const [reason, setReason] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reason.trim() || reason.trim().length < 5) {
      setError('Reopen reason must be at least 5 characters long.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const updated = await updateCaseStatus(caseId, {
        target_status: 'IN_REVIEW',
        reason: reason.trim(),
      });
      onSuccess(updated);
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.statusCode === 403) {
          setError('Permission Denied: You are not authorized to reopen this case.');
        } else if (err.statusCode === 409) {
          setError(err.message || 'Conflict: Only RESOLVED or CLOSED cases can be reopened.');
        } else if (err.statusCode === 422) {
          setError(err.message || 'Validation Error: Reopen reason must be at least 5 characters.');
        } else {
          setError(err.message || 'Failed to reopen case.');
        }
      } else {
        const msg = err instanceof Error ? err.message : 'Unknown error reopening case';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="reopen-case-title">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <RotateCcw size={20} className="modal-icon-info" />
            <h3 id="reopen-case-title" className="modal-title">
              Reopen Case Investigation
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
              <label htmlFor="reopen-reason" className="form-label">
                Reopening Justification <span className="required-star">*</span>
              </label>
              <textarea
                id="reopen-reason"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Explain why this investigation is being reopened (e.g. new chargeback evidence, customer dispute appeal)..."
                className="form-textarea"
                rows={4}
                disabled={loading}
                required
              />
              <span className="field-hint">
                This will reset status to IN_REVIEW, clear previous disposition, and reactivate active investigation.
              </span>
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-modal-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn-modal-primary" disabled={loading || reason.trim().length < 5}>
              {loading ? 'Reopening...' : 'Reopen Case'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
