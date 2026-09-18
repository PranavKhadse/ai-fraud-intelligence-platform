import React, { useState, useEffect, useCallback } from 'react';
import {
  ArrowLeft,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Clock,
  MapPin,
  DollarSign,
  Layers,
  FileText,
  Activity,
  ArrowRightLeft,
  ChevronDown,
  ChevronRight,
  Sliders,
  UserCheck,
  UserX,
  UserPlus,
  RotateCcw,
  Send,
  MessageSquare,
  History,
  RefreshCw,
  XCircle,
} from 'lucide-react';

import { Badge } from '../common/Badge.tsx';
import { RiskScoreMeter } from '../common/RiskScoreMeter.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { ShapWaterfall } from '../explainability/ShapWaterfall.tsx';
import { DispositionModal } from './DispositionModal.tsx';
import { AssignModal, EscalateModal, CloseModal, ReopenModal } from './CaseActionModals.tsx';
import {
  fetchCaseDetail,
  fetchCaseTimeline,
  createCaseNote,
  updateCaseAssignment,
} from '../../api/caseApi.ts';
import { ApiClientError } from '../../api/client.ts';
import type {
  CaseDetailResponse,
  CaseNoteItem,
  CaseNoteType,
  CasePriority,
  CaseResponse,
  CaseStatus,
  CaseTimelineResponse,
  TimelineEventItem,
} from '../../types/case.ts';
import type {
  DecisionAction,
  ReasonSeverity,
  RuleOutcome,
} from '../../types/api.ts';


interface CaseInvestigationWorkspaceProps {
  caseId: string;
  onBackToQueue: () => void;
}

// 7 Feature Category Mappings from Phase 3 Feature Engine
const FEATURE_CATEGORY_MAP: Record<string, string[]> = {
  'Velocity & Frequency': [
    'txn_count_1h',
    'txn_count_6h',
    'txn_count_24h',
    'txn_count_7d',
    'txn_count_30d',
    'time_since_prev_txn_seconds',
    'is_first_account_txn',
  ],
  'Spending & Monetary Volume': [
    'amt_sum_1h',
    'amt_sum_24h',
    'amt_sum_7d',
    'amt_sum_30d',
    'amt_mean_24h',
    'amt_mean_7d',
    'amt_max_24h',
    'amt_median_30d',
  ],
  'Spending Deviation & Z-Scores': [
    'historical_amount_mean',
    'historical_amount_std',
    'historical_amount_median',
    'amount_zscore',
    'amount_ratio_to_historical_mean',
  ],
  'Account History & Diversity': [
    'account_txn_count_before',
    'account_total_spend_before',
    'account_avg_amount_before',
    'account_max_amount_before',
    'account_unique_merchant_count_before',
    'account_unique_category_count_before',
  ],
  'Merchant & Category Interactions': [
    'account_merchant_txn_count_before',
    'account_category_txn_count_before',
    'account_merchant_spend_before',
    'account_category_spend_before',
    'merchant_txn_count_before',
    'category_txn_count_before',
  ],
  'Geographic & Impossible Travel': [
    'cardholder_merchant_distance_km',
    'distance_from_prev_merchant_km',
    'implied_travel_speed_kmh',
    'is_impossible_travel_speed',
  ],
  'Temporal & Periodicity': [
    'transaction_hour',
    'day_of_week',
    'day_of_month',
    'month',
    'week_of_year',
    'is_weekend',
    'is_night',
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
  ],
};

export const CaseInvestigationWorkspace: React.FC<CaseInvestigationWorkspaceProps> = ({
  caseId,
  onBackToQueue,
}) => {
  const [detail, setDetail] = useState<CaseDetailResponse | null>(null);
  const [timeline, setTimeline] = useState<CaseTimelineResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [timelineLoading, setTimelineLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // Active Tab for Column 3: 'notes' | 'timeline'
  const [activityTab, setActivityTab] = useState<'notes' | 'timeline'>('notes');

  // New Note Composer State
  const [newNoteContent, setNewNoteContent] = useState<string>('');
  const [newNoteType, setNewNoteType] = useState<CaseNoteType>('INVESTIGATION');
  const [submittingNote, setSubmittingNote] = useState<boolean>(false);

  // Modal Dialog States
  const [isDispositionOpen, setIsDispositionOpen] = useState<boolean>(false);
  const [isAssignOpen, setIsAssignOpen] = useState<boolean>(false);
  const [isEscalateOpen, setIsEscalateOpen] = useState<boolean>(false);
  const [isCloseOpen, setIsCloseOpen] = useState<boolean>(false);
  const [isReopenOpen, setIsReopenOpen] = useState<boolean>(false);

  // Feature Group Accordion State
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({
    'Velocity & Frequency': true,
    'Spending & Monetary Volume': true,
    'Spending Deviation & Z-Scores': true,
    'Geographic & Impossible Travel': true,
  });

  const loadCaseData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setActionError(null);

    try {
      const [caseData, timelineData] = await Promise.all([
        fetchCaseDetail(caseId),
        fetchCaseTimeline(caseId).catch(() => null),
      ]);
      setDetail(caseData);
      setTimeline(timelineData);
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.statusCode === 403) {
          setError('Permission Denied: You do not have permission to view this case.');
        } else if (err.statusCode === 404) {
          setError(`Case '${caseId}' was not found in the database.`);
        } else {
          setError(err.message || 'Failed to load case investigation details.');
        }
      } else {
        const msg = err instanceof Error ? err.message : 'Unknown error loading case';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    loadCaseData();
  }, [loadCaseData]);

  const refreshTimelineOnly = useCallback(async () => {
    setTimelineLoading(true);
    try {
      const timelineData = await fetchCaseTimeline(caseId);
      setTimeline(timelineData);
    } catch {
      // Timeline refresh error handled quietly
    } finally {
      setTimelineLoading(false);
    }
  }, [caseId]);

  // Quick Assignment Actions
  const handleClaim = async () => {
    setActionError(null);
    try {
      const updated = await updateCaseAssignment(caseId, { action: 'CLAIM' });
      setDetail((prev) => (prev ? { ...prev, case: updated } : null));
      refreshTimelineOnly();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to claim case';
      setActionError(msg);
    }
  };

  const handleUnassign = async () => {
    setActionError(null);
    try {
      const updated = await updateCaseAssignment(caseId, { action: 'UNASSIGN' });
      setDetail((prev) => (prev ? { ...prev, case: updated } : null));
      refreshTimelineOnly();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to unassign case';
      setActionError(msg);
    }
  };

  // Add Note Handler
  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newNoteContent.trim() || newNoteContent.trim().length < 2) {
      return;
    }

    setSubmittingNote(true);
    setActionError(null);

    try {
      const noteItem = await createCaseNote(caseId, {
        content: newNoteContent.trim(),
        note_type: newNoteType,
      });
      setDetail((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          notes: [...prev.notes, noteItem],
          notes_total: prev.notes_total + 1,
        };
      });
      setNewNoteContent('');
      refreshTimelineOnly();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to append note';
      setActionError(msg);
    } finally {
      setSubmittingNote(false);
    }
  };

  const toggleCategory = (catName: string) => {
    setExpandedCategories((prev) => ({
      ...prev,
      [catName]: !prev[catName],
    }));
  };

  const formatCurrency = (val: number, currency: string = 'USD'): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency || 'USD',
      minimumFractionDigits: 2,
    }).format(val);
  };

  const formatTimestamp = (isoString?: string | null): string => {
    if (!isoString) return '—';
    try {
      return new Date(isoString).toLocaleString('en-US', {
        dateStyle: 'medium',
        timeStyle: 'medium',
        hour12: false,
      });
    } catch {
      return isoString;
    }
  };

  const formatValue = (val: unknown): string => {
    if (val === null || val === undefined) return 'null';
    if (typeof val === 'number') {
      return Number.isInteger(val) ? val.toString() : val.toFixed(4);
    }
    if (typeof val === 'boolean') {
      return val ? 'True' : 'False';
    }
    return String(val);
  };

  const renderStatusBadge = (status: CaseStatus) => {
    switch (status) {
      case 'OPEN':
        return <Badge variant="review">OPEN</Badge>;
      case 'IN_REVIEW':
        return <Badge variant="info">IN REVIEW</Badge>;
      case 'ESCALATED':
        return <Badge variant="block">ESCALATED</Badge>;
      case 'RESOLVED':
        return <Badge variant="approve">RESOLVED</Badge>;
      case 'CLOSED':
        return <Badge variant="neutral">CLOSED</Badge>;
    }
  };

  const renderPriorityBadge = (priority: CasePriority) => {
    switch (priority) {
      case 'CRITICAL':
        return <Badge variant="block">CRITICAL PRIORITY</Badge>;
      case 'HIGH':
        return <Badge variant="block">HIGH PRIORITY</Badge>;
      case 'MEDIUM':
        return <Badge variant="review">MEDIUM PRIORITY</Badge>;
      case 'LOW':
        return <Badge variant="approve">LOW PRIORITY</Badge>;
    }
  };

  const renderDecisionBadge = (action: DecisionAction) => {
    switch (action) {
      case 'APPROVE':
        return <Badge variant="approve" icon={<CheckCircle2 size={12} />}>APPROVE</Badge>;
      case 'REVIEW':
        return <Badge variant="review" icon={<AlertTriangle size={12} />}>REVIEW</Badge>;
      case 'BLOCK':
        return <Badge variant="block" icon={<ShieldAlert size={12} />}>BLOCK</Badge>;
    }
  };

  const renderOutcomeBadge = (outcome: RuleOutcome) => {
    switch (outcome) {
      case 'APPROVE':
        return <Badge variant="approve">APPROVE</Badge>;
      case 'REVIEW':
        return <Badge variant="review">REVIEW</Badge>;
      case 'BLOCK':
        return <Badge variant="block">BLOCK</Badge>;
      case 'MONITOR':
        return <Badge variant="neutral">MONITOR</Badge>;
    }
  };

  const renderSeverityBadge = (severity: ReasonSeverity) => {
    switch (severity) {
      case 'CRITICAL':
      case 'HIGH':
        return <Badge variant="block">{severity}</Badge>;
      case 'MEDIUM':
        return <Badge variant="review">{severity}</Badge>;
      case 'LOW':
        return <Badge variant="approve">{severity}</Badge>;
      case 'INFO':
        return <Badge variant="neutral">{severity}</Badge>;
    }
  };


  if (loading && !detail) {
    return (
      <div className="workspace-loading-wrap">
        <LoadingSpinner message="Loading full investigation workspace and persisted evidence..." />
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="workspace-error-wrap">
        <ErrorBanner
          title="Case Investigation Unavailable"
          message={error || 'Case could not be found'}
          onRetry={loadCaseData}
        />
        <div style={{ marginTop: '1rem' }}>
          <button type="button" className="btn-secondary" onClick={onBackToQueue}>
            <ArrowLeft size={14} />
            <span>Return to Review Queue</span>
          </button>
        </div>
      </div>
    );
  }

  const { case: c, transaction: tx, evaluation: ev, notes } = detail;
  const features = (tx.features_snapshot as Record<string, unknown>) || {};
  const isResolvedOrClosed = c.status === 'RESOLVED' || c.status === 'CLOSED';

  return (
    <div className="workspace-container">
      {/* Top Navigation & Case Status Banner */}
      <header className="workspace-header">
        <div className="workspace-header-top">
          <button type="button" className="btn-back-queue" onClick={onBackToQueue}>
            <ArrowLeft size={16} />
            <span>Review Queue</span>
          </button>

          <div className="case-title-row">
            <h1 className="case-number-heading font-mono">{c.case_number}</h1>
            {renderPriorityBadge(c.priority)}
            {renderStatusBadge(c.status)}
            <Badge variant="neutral">{c.trigger_source}</Badge>
          </div>

          <div className="workspace-header-actions">
            <button
              type="button"
              className="btn-workspace-action"
              onClick={loadCaseData}
              title="Refresh case detail"
            >
              <RefreshCw size={14} />
              <span>Refresh</span>
            </button>

            {/* Assignment Actions */}
            {!isResolvedOrClosed && (
              <>
                {!c.assigned_to ? (
                  <button
                    type="button"
                    className="btn-workspace-action primary"
                    onClick={handleClaim}
                    title="Claim case directly to yourself"
                  >
                    <UserCheck size={14} />
                    <span>Claim Case</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn-workspace-action"
                    onClick={handleUnassign}
                    title="Release assignment"
                  >
                    <UserX size={14} />
                    <span>Unassign</span>
                  </button>
                )}

                <button
                  type="button"
                  className="btn-workspace-action"
                  onClick={() => setIsAssignOpen(true)}
                  title="Assign to specific analyst"
                >
                  <UserPlus size={14} />
                  <span>Assign...</span>
                </button>
              </>
            )}

            {/* Lifecycle Status Actions */}
            {(c.status === 'OPEN' || c.status === 'IN_REVIEW') && (
              <button
                type="button"
                className="btn-workspace-action warning"
                onClick={() => setIsEscalateOpen(true)}
                title="Escalate to senior tier triage"
              >
                <AlertTriangle size={14} />
                <span>Escalate</span>
              </button>
            )}

            {!isResolvedOrClosed && (
              <>
                <button
                  type="button"
                  className="btn-workspace-action success"
                  onClick={() => setIsDispositionOpen(true)}
                  title="Submit definitive review disposition"
                >
                  <CheckCircle2 size={14} />
                  <span>Submit Disposition</span>
                </button>

                <button
                  type="button"
                  className="btn-workspace-action"
                  onClick={() => setIsCloseOpen(true)}
                  title="Close case"
                >
                  <XCircle size={14} />
                  <span>Close Case</span>
                </button>
              </>
            )}

            {isResolvedOrClosed && (
              <button
                type="button"
                className="btn-workspace-action primary"
                onClick={() => setIsReopenOpen(true)}
                title="Reopen closed/resolved case"
              >
                <RotateCcw size={14} />
                <span>Reopen Case</span>
              </button>
            )}
          </div>
        </div>

        {/* Case Metadata Subtitle Bar */}
        <div className="workspace-header-meta">
          <div className="meta-item">
            <span className="meta-label">Assigned:</span>
            <span className="meta-value font-mono">
              {c.assigned_to ? c.assigned_to : <em className="text-muted">Unassigned</em>}
            </span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Opened:</span>
            <span className="meta-value font-mono">{formatTimestamp(c.opened_at)}</span>
          </div>
          {c.disposition && (
            <div className="meta-item">
              <span className="meta-label">Disposition:</span>
              <span className="meta-value font-mono text-approve">
                {c.disposition} ({c.dispositioned_by || 'analyst'})
              </span>
            </div>
          )}
          {c.closed_at && (
            <div className="meta-item">
              <span className="meta-label">Closed:</span>
              <span className="meta-value font-mono">{formatTimestamp(c.closed_at)}</span>
            </div>
          )}
        </div>
      </header>

      {/* Action Error Banner */}
      {actionError && (
        <ErrorBanner
          title="Case Action Error"
          message={actionError}
          onRetry={() => setActionError(null)}
        />
      )}

      {/* 3-Column Responsive Workspace Grid */}
      <div className="workspace-grid-layout">
        {/* ===================================================================
            COLUMN 1: Canonical Transaction Context & 55-Feature Snapshot
           =================================================================== */}
        <div className="workspace-column column-context">
          <div className="column-header">
            <DollarSign size={16} />
            <h2>Transaction Context</h2>
          </div>

          <div className="workspace-card">
            <div className="metadata-grid">
              <div className="metadata-card">
                <span className="meta-label">Transaction Amount</span>
                <span className="meta-val highlight">
                  {formatCurrency(tx.amount, tx.currency)}
                </span>
              </div>

              <div className="metadata-card">
                <span className="meta-label">Cardholder Account</span>
                <span className="meta-val font-mono">{tx.account_id}</span>
              </div>

              <div className="metadata-card">
                <span className="meta-label">Merchant Category</span>
                <span className="meta-val">{tx.merchant_category}</span>
              </div>

              <div className="metadata-card">
                <span className="meta-label">Job Category</span>
                <span className="meta-val">{tx.job_category}</span>
              </div>

              <div className="metadata-card">
                <span className="meta-label">Cardholder Coords</span>
                <span className="meta-val font-mono text-xs">
                  <MapPin size={11} /> {tx.cardholder_lat.toFixed(4)}, {tx.cardholder_long.toFixed(4)}
                </span>
              </div>

              <div className="metadata-card">
                <span className="meta-label">Merchant Coords</span>
                <span className="meta-val font-mono text-xs">
                  <MapPin size={11} /> {tx.merchant_lat.toFixed(4)}, {tx.merchant_long.toFixed(4)}
                </span>
              </div>

              <div className="metadata-card">
                <span className="meta-label">City Population</span>
                <span className="meta-val font-mono">
                  {tx.city_pop ? tx.city_pop.toLocaleString() : '—'}
                </span>
              </div>

              <div className="metadata-card">
                <span className="meta-label">Timestamp</span>
                <span className="meta-val font-mono text-xs">
                  <Clock size={11} /> {formatTimestamp(tx.transaction_timestamp)}
                </span>
              </div>
            </div>
          </div>

          {/* 55-Feature Categorized Snapshot */}
          <div className="workspace-card" style={{ marginTop: '1rem' }}>
            <div className="section-title">
              <Layers size={16} />
              <span>55-Feature Point-in-Time Snapshot</span>
              <span className="section-badge">{Object.keys(features).length} Features</span>
            </div>

            <div className="feature-groups-accordion">
              {Object.entries(FEATURE_CATEGORY_MAP).map(([catTitle, featureNames]) => {
                const groupFeatures = featureNames.filter((fn) => fn in features);
                if (groupFeatures.length === 0) return null;
                const isExpanded = expandedCategories[catTitle] ?? false;

                return (
                  <div key={catTitle} className="feature-group-container">
                    <button
                      type="button"
                      className="feature-group-header"
                      onClick={() => toggleCategory(catTitle)}
                      aria-expanded={isExpanded}
                    >
                      <div className="group-header-left">
                        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        <span className="group-title">{catTitle}</span>
                      </div>
                      <span className="group-count font-mono text-muted">
                        {groupFeatures.length}
                      </span>
                    </button>

                    {isExpanded && (
                      <div className="feature-group-body">
                        <div className="feature-grid">
                          {groupFeatures.map((fn) => (
                            <div key={fn} className="feature-pill">
                              <span className="feat-name font-mono">{fn}</span>
                              <span className="feat-val font-mono">{formatValue(features[fn])}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* ===================================================================
            COLUMN 2: Risk Intelligence & Local TreeSHAP Explainability
           =================================================================== */}
        <div className="workspace-column column-risk">
          <div className="column-header">
            <Activity size={16} />
            <h2>Risk Intelligence & Explainability</h2>
          </div>

          {ev ? (
            <>
              {/* Risk Decision Summary Card */}
              <div className="workspace-card">
                <div className="risk-assessment-grid">
                  <div className="assessment-item score-item">
                    <span className="item-label">Calibrated Risk Score</span>
                    <div className="score-meter-wrap">
                      <RiskScoreMeter score={ev.risk_score} tier={ev.risk_tier} />
                    </div>
                  </div>

                  <div className="assessment-item">
                    <span className="item-label">Model Probability</span>
                    <span className="item-value font-mono">
                      {(ev.model_score * 100).toFixed(2)}%{' '}
                      <small className="text-muted">({ev.model_score.toFixed(6)})</small>
                    </span>
                  </div>

                  <div className="assessment-item">
                    <span className="item-label">Policy Decision</span>
                    <div className="badge-row">
                      {renderDecisionBadge(ev.decision_action)}
                      {ev.is_overridden && (
                        <Badge variant="review" icon={<ArrowRightLeft size={11} />}>
                          RULE OVERRIDE
                        </Badge>
                      )}
                    </div>
                  </div>

                  <div className="assessment-item">
                    <span className="item-label">Model Version</span>
                    <span className="item-value font-mono">
                      v{ev.model_version} ({ev.policy_mode})
                    </span>
                  </div>
                </div>

                <div className="decision-reason-box" style={{ marginTop: '0.75rem' }}>
                  <strong>Decision Rationale:</strong>
                  <p>{ev.decision_reason}</p>
                </div>
              </div>

              {/* TreeSHAP Decision Waterfall Component */}
              <div className="workspace-card" style={{ marginTop: '1rem' }}>
                <ShapWaterfall
                  baseValue={ev.base_value}
                  outputMargin={ev.output_margin}
                  modelScore={ev.model_score}
                  featureAttributions={ev.feature_attributions}
                />
              </div>

              {/* Plain-English Reason Codes */}
              {ev.reason_codes.length > 0 && (
                <div className="workspace-card" style={{ marginTop: '1rem' }}>
                  <div className="section-title">
                    <FileText size={16} />
                    <span>Plain-English Reason Codes</span>
                    <span className="section-badge">{ev.reason_codes.length} Codes</span>
                  </div>

                  <div className="reason-cards-list">
                    {ev.reason_codes.map((rc) => (
                      <div key={rc.code} className="reason-code-card">
                        <div className="reason-card-header">
                          <div className="reason-headline-group">
                            <span className="reason-rank">#{rc.rank}</span>
                            <span className="reason-headline">{rc.headline}</span>
                          </div>
                          <div className="reason-badges">
                            <Badge variant="neutral">{rc.category}</Badge>
                            {renderSeverityBadge(rc.severity)}
                          </div>
                        </div>
                        <p className="reason-description">{rc.description}</p>
                        <div className="reason-code-tag">
                          Code: <span className="font-mono">{rc.code}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Triggered Business Rules */}
              {ev.rule_matches.length > 0 && (
                <div className="workspace-card" style={{ marginTop: '1rem' }}>
                  <div className="section-title">
                    <Sliders size={16} />
                    <span>Triggered Business Rules</span>
                    <span className="section-badge">{ev.rule_matches.length} Triggered</span>
                  </div>

                  <div className="rule-cards-list">
                    {ev.rule_matches.map((rm) => (
                      <div key={rm.rule_id} className="rule-match-card">
                        <div className="rule-header">
                          <div className="rule-id-wrap">
                            <span className="rule-id font-mono">{rm.rule_id}</span>
                            <span className="rule-priority">Priority #{rm.priority}</span>
                          </div>
                          <div className="rule-outcome-wrap">
                            {renderOutcomeBadge(rm.outcome)}
                            <Badge variant="neutral">{rm.rule_type}</Badge>
                          </div>
                        </div>
                        <p className="rule-desc">{rm.description}</p>
                        <div className="rule-condition font-mono">
                          Condition: <code>{rm.feature_name} {rm.operator} {rm.comparison_value}</code>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="empty-sub-section">
              <AlertTriangle size={16} />
              <span>No risk evaluation context persisted for this transaction.</span>
            </div>
          )}
        </div>

        {/* ===================================================================
            COLUMN 3: Case Activity (Investigation Notes & Audit Timeline)
           =================================================================== */}
        <div className="workspace-column column-activity">
          <div className="column-header-tabs">
            <button
              type="button"
              className={`activity-tab-btn ${activityTab === 'notes' ? 'active' : ''}`}
              onClick={() => setActivityTab('notes')}
            >
              <MessageSquare size={15} />
              <span>Investigation Notes ({notes.length})</span>
            </button>
            <button
              type="button"
              className={`activity-tab-btn ${activityTab === 'timeline' ? 'active' : ''}`}
              onClick={() => {
                setActivityTab('timeline');
                if (!timeline) refreshTimelineOnly();
              }}
            >
              <History size={15} />
              <span>Audit Timeline ({timeline?.events.length ?? 0})</span>
            </button>
          </div>

          {/* TAB 1: Investigation Notes Stream */}
          {activityTab === 'notes' && (
            <div className="notes-stream-panel">
              <div className="notes-scroll-area">
                {notes.length === 0 ? (
                  <div className="empty-sub-section">
                    <span>No notes recorded for this case yet.</span>
                  </div>
                ) : (
                  notes.map((n: CaseNoteItem) => (
                    <div key={n.id} className="case-note-bubble">
                      <div className="note-bubble-header">
                        <div className="author-info">
                          <Badge
                            variant={
                              n.author_role === 'ADMIN'
                                ? 'block'
                                : n.author_role === 'ANALYST'
                                ? 'approve'
                                : 'neutral'
                            }
                          >
                            {n.author_role}
                          </Badge>
                          <span className="author-id font-mono">{n.author_id}</span>
                          <span className="note-type-chip">{n.note_type}</span>
                        </div>
                        <span className="note-timestamp font-mono">
                          {formatTimestamp(n.created_at)}
                        </span>
                      </div>
                      <div className="note-bubble-content">
                        <p>{n.content}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Append-Only Note Composer */}
              <form onSubmit={handleAddNote} className="note-composer-box">
                <div className="composer-header">
                  <span className="composer-label">Append Investigation Note</span>
                  <select
                    value={newNoteType}
                    onChange={(e) => setNewNoteType(e.target.value as CaseNoteType)}
                    className="composer-type-select"
                    disabled={submittingNote}
                  >
                    <option value="INVESTIGATION">Investigation</option>
                    <option value="ESCALATION">Escalation</option>
                    <option value="DISPOSITION">Disposition</option>
                    <option value="SYSTEM_AUDIT">System Audit</option>
                  </select>
                </div>

                <textarea
                  value={newNoteContent}
                  onChange={(e) => setNewNoteContent(e.target.value)}
                  placeholder="Record investigation findings, evidence, customer interaction notes..."
                  className="composer-textarea"
                  rows={3}
                  disabled={submittingNote}
                />

                <div className="composer-footer">
                  <span className="composer-hint">
                    Notes are append-only and stamped with your actor identity.
                  </span>
                  <button
                    type="submit"
                    className="btn-send-note"
                    disabled={submittingNote || !newNoteContent.trim()}
                  >
                    <Send size={13} />
                    <span>{submittingNote ? 'Saving...' : 'Add Note'}</span>
                  </button>
                </div>
              </form>
            </div>
          )}

          {/* TAB 2: Case Audit Timeline Stream */}
          {activityTab === 'timeline' && (
            <div className="timeline-stream-panel">
              {timelineLoading ? (
                <LoadingSpinner message="Querying case audit timeline events..." />
              ) : timeline && timeline.events.length > 0 ? (
                <div className="audit-timeline">
                  {timeline.events.map((evt: TimelineEventItem) => (
                    <div key={evt.id} className="audit-timeline-item">
                      <div className="timeline-marker" />
                      <div className="timeline-content">
                        <div className="timeline-header">
                          <span className="audit-event-type font-mono">{evt.event_type}</span>
                          <span className="audit-action-badge">{evt.action}</span>
                          <span className="audit-time font-mono">
                            {formatTimestamp(evt.event_timestamp)}
                          </span>
                        </div>
                        <div className="timeline-details">
                          <span className="audit-actor">
                            Actor: <strong>{evt.actor_type}</strong> ({evt.actor_id || 'system'})
                          </span>
                          {evt.correlation_id && (
                            <span className="audit-corr font-mono">
                              Corr: {evt.correlation_id}
                            </span>
                          )}
                        </div>
                        {evt.payload && Object.keys(evt.payload).length > 0 && (
                          <div className="timeline-payload font-mono text-xs">
                            <pre>{JSON.stringify(evt.payload, null, 2)}</pre>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-sub-section">
                  <span>No audit events recorded for this case entity.</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Action Modals */}
      {isDispositionOpen && (
        <DispositionModal
          isOpen={isDispositionOpen}
          onClose={() => setIsDispositionOpen(false)}
          caseId={c.id}
          caseNumber={c.case_number}
          onSuccess={(updated: CaseResponse) => {
            setDetail((prev) => (prev ? { ...prev, case: updated } : null));
            refreshTimelineOnly();
          }}
        />
      )}

      {isAssignOpen && (
        <AssignModal
          isOpen={isAssignOpen}
          onClose={() => setIsAssignOpen(false)}
          caseId={c.id}
          caseNumber={c.case_number}
          currentAssignee={c.assigned_to}
          onSuccess={(updated: CaseResponse) => {
            setDetail((prev) => (prev ? { ...prev, case: updated } : null));
            refreshTimelineOnly();
          }}
        />
      )}

      {isEscalateOpen && (
        <EscalateModal
          isOpen={isEscalateOpen}
          onClose={() => setIsEscalateOpen(false)}
          caseId={c.id}
          caseNumber={c.case_number}
          onSuccess={(updated: CaseResponse) => {
            setDetail((prev) => (prev ? { ...prev, case: updated } : null));
            refreshTimelineOnly();
          }}
        />
      )}

      {isCloseOpen && (
        <CloseModal
          isOpen={isCloseOpen}
          onClose={() => setIsCloseOpen(false)}
          caseId={c.id}
          caseNumber={c.case_number}
          onSuccess={(updated: CaseResponse) => {
            setDetail((prev) => (prev ? { ...prev, case: updated } : null));
            refreshTimelineOnly();
          }}
        />
      )}

      {isReopenOpen && (
        <ReopenModal
          isOpen={isReopenOpen}
          onClose={() => setIsReopenOpen(false)}
          caseId={c.id}
          caseNumber={c.case_number}
          onSuccess={(updated: CaseResponse) => {
            setDetail((prev) => (prev ? { ...prev, case: updated } : null));
            refreshTimelineOnly();
          }}
        />
      )}
    </div>
  );
};
