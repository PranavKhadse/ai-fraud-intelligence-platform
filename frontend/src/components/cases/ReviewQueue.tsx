import React, { useState, useCallback, useEffect } from 'react';
import {
  Inbox,
  AlertOctagon,
  Clock,
  UserCheck,
  CheckCircle2,
  ShieldAlert,
  ArrowUpDown,
  Search,
  RefreshCw,
  Plus,
  ChevronLeft,
  ChevronRight,
  User,
} from 'lucide-react';

import { StatCard } from '../common/StatCard.tsx';
import { Badge } from '../common/Badge.tsx';
import { RiskScoreMeter } from '../common/RiskScoreMeter.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { EmptyState } from '../common/EmptyState.tsx';
import { CreateCaseModal } from './CreateCaseModal.tsx';
import { fetchCases, fetchCaseSummary, updateCaseAssignment } from '../../api/caseApi.ts';
import { usePolling } from '../../hooks/usePolling.ts';
import type {
  CaseListResponse,
  CasePriority,
  CaseQueueItem,
  CaseQueryParams,
  CaseStatus,
  CaseSummaryResponse,
} from '../../types/case.ts';
import type { DecisionAction, RiskTier } from '../../types/api.ts';

export interface QueueFiltersState {
  status?: CaseStatus;
  priority?: CasePriority;
  assigned_to?: string;
  risk_tier?: RiskTier;
  search_term: string;
  sort_by: 'opened_at' | 'priority' | 'risk_score';
  sort_order: 'desc' | 'asc';
  limit: number;
  page: number;
}

interface ReviewQueueProps {
  onSelectCase: (caseId: string) => void;
  initialFilters?: Partial<QueueFiltersState>;
  onFiltersChange?: (filters: QueueFiltersState) => void;
}

export const ReviewQueue: React.FC<ReviewQueueProps> = ({
  onSelectCase,
  initialFilters,
  onFiltersChange,
}) => {
  const [filters, setFilters] = useState<QueueFiltersState>({
    status: initialFilters?.status,
    priority: initialFilters?.priority,
    assigned_to: initialFilters?.assigned_to,
    risk_tier: initialFilters?.risk_tier,
    search_term: initialFilters?.search_term || '',
    sort_by: initialFilters?.sort_by || 'opened_at',
    sort_order: initialFilters?.sort_order || 'desc',
    limit: initialFilters?.limit || 20,
    page: initialFilters?.page || 1,
  });

  const [summary, setSummary] = useState<CaseSummaryResponse | null>(null);

  const [isCreateModalOpen, setIsCreateModalOpen] = useState<boolean>(false);
  const [createTxIdInput, setCreateTxIdInput] = useState<string>('');
  const [actionError, setActionError] = useState<string | null>(null);

  // Sync state to parent filter store for back-navigation preservation
  useEffect(() => {
    if (onFiltersChange) {
      onFiltersChange(filters);
    }
  }, [filters, onFiltersChange]);

  // Summary Metrics Fetch
  const loadSummary = useCallback(async () => {
    try {
      const res = await fetchCaseSummary();
      setSummary(res);
    } catch {
      // Summary error handled quietly in background
    }
  }, []);

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  // Query Params for Queue Items
  const queryParams: CaseQueryParams = {
    limit: filters.limit,
    offset: (filters.page - 1) * filters.limit,
    status: filters.status,
    priority: filters.priority,
    assigned_to: filters.assigned_to,
    risk_tier: filters.risk_tier,
    search_term: filters.search_term.trim() || undefined,
    sort_by: filters.sort_by,
    sort_order: filters.sort_order,
  };

  const fetchFn = useCallback(() => {
    return fetchCases(queryParams);
  }, [
    filters.limit,
    filters.page,
    filters.status,
    filters.priority,
    filters.assigned_to,
    filters.risk_tier,
    filters.search_term,
    filters.sort_by,
    filters.sort_order,
  ]);

  const {
    data,
    loading,
    isPolling,
    error,
    isAutoPolling,
    toggleAutoPolling,
    refresh,
  } = usePolling<CaseListResponse>(fetchFn, {
    interval: 10000,
    enabled: true,
  });


  const handleManualRefresh = async () => {
    setActionError(null);
    await Promise.all([refresh(), loadSummary()]);
  };

  const handleClaim = async (e: React.MouseEvent, caseId: string) => {
    e.stopPropagation();
    setActionError(null);
    try {
      await updateCaseAssignment(caseId, { action: 'CLAIM' });
      await handleManualRefresh();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to claim case';
      setActionError(msg);
    }
  };

  const handleSort = (field: 'opened_at' | 'priority' | 'risk_score') => {
    setFilters((prev) => {
      if (prev.sort_by === field) {
        return {
          ...prev,
          sort_order: prev.sort_order === 'desc' ? 'asc' : 'desc',
          page: 1,
        };
      }
      return {
        ...prev,
        sort_by: field,
        sort_order: 'desc',
        page: 1,
      };
    });
  };

  const totalPages = data ? Math.max(1, Math.ceil(data.total / filters.limit)) : 1;

  const formatCurrency = (val: number, currency: string = 'USD'): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency || 'USD',
      minimumFractionDigits: 2,
    }).format(val);
  };

  const formatDateTime = (isoString?: string | null): string => {
    if (!isoString) return '—';
    try {
      const d = new Date(isoString);
      return d.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  const renderPriorityBadge = (priority: CasePriority) => {
    switch (priority) {
      case 'CRITICAL':
        return <Badge variant="block">CRITICAL</Badge>;
      case 'HIGH':
        return <Badge variant="block">HIGH</Badge>;
      case 'MEDIUM':
        return <Badge variant="review">MEDIUM</Badge>;
      case 'LOW':
        return <Badge variant="approve">LOW</Badge>;
    }
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

  const renderDecisionBadge = (action: DecisionAction) => {
    switch (action) {
      case 'APPROVE':
        return <Badge variant="approve">APPROVE</Badge>;
      case 'REVIEW':
        return <Badge variant="review">REVIEW</Badge>;
      case 'BLOCK':
        return <Badge variant="block">BLOCK</Badge>;
    }
  };

  return (
    <div className="review-queue-container">
      {/* Header & Escalation Trigger */}
      <header className="queue-header">
        <div>
          <h1 className="dashboard-headline">Human Review Queue</h1>
          <p className="dashboard-subheadline">
            Operational triage, case assignments, risk investigations, and authoritative dispositions
          </p>
        </div>

        <div className="queue-header-actions">
          <button
            type="button"
            className="btn-create-case"
            onClick={() => {
              const tx = window.prompt('Enter Transaction ID / UUID to manually escalate:');
              if (tx && tx.trim()) {
                setCreateTxIdInput(tx.trim());
                setIsCreateModalOpen(true);
              }
            }}
          >
            <Plus size={16} />
            <span>Escalate Transaction to Case</span>
          </button>
        </div>
      </header>

      {/* KPI Summary Cards Grid */}
      <section className="queue-summary-section" aria-label="Review Queue Metrics">
        <div className="stats-grid">
          <StatCard
            title="Total Open Cases"
            value={summary ? summary.total_open.toLocaleString() : '—'}
            icon={<Inbox size={18} />}
            sublabel="Awaiting Resolution"
            subvalue={`${summary ? summary.in_review_count : 0} in active review`}
            tone="primary"
          />

          <StatCard
            title="Unassigned Backlog"
            value={summary ? summary.unassigned_count.toLocaleString() : '—'}
            icon={<AlertOctagon size={18} style={{ color: 'var(--status-review)' }} />}
            sublabel="Needs Assignment"
            subvalue={summary && summary.unassigned_count > 0 ? 'Triage required' : 'Queue clear'}
            tone="review"
            badge={<Badge variant="review">UNASSIGNED</Badge>}
          />

          <StatCard
            title="Escalated Cases"
            value={summary ? summary.escalated_count.toLocaleString() : '—'}
            icon={<ShieldAlert size={18} style={{ color: 'var(--status-block)' }} />}
            sublabel="Senior Tier Triage"
            subvalue="Elevated priority"
            tone="block"
            badge={<Badge variant="block">ESCALATED</Badge>}
          />

          <StatCard
            title="Critical Priority"
            value={summary ? summary.critical_priority_count.toLocaleString() : '—'}
            icon={<AlertOctagon size={18} style={{ color: 'var(--status-block)' }} />}
            sublabel="Urgent Attention"
            subvalue="SLA priority"
            tone="block"
          />

          <StatCard
            title="Resolved (Today)"
            value={summary ? summary.resolved_today.toLocaleString() : '—'}
            icon={<CheckCircle2 size={18} style={{ color: 'var(--status-approve)' }} />}
            sublabel="Last 24 Hours"
            subvalue={`${summary ? summary.resolved_last_24h : 0} total 24h`}
            tone="approve"
            badge={<Badge variant="approve">RESOLVED</Badge>}
          />
        </div>
      </section>

      {/* Action Error Banner */}
      {actionError && (
        <ErrorBanner
          title="Case Action Error"
          message={actionError}
          onRetry={() => setActionError(null)}
        />
      )}

      {/* Queue Filter and Controls Toolbar */}
      <section className="queue-controls-bar">
        <div className="queue-search-wrap">
          <Search size={15} className="search-icon" />
          <input
            type="text"
            value={filters.search_term}
            onChange={(e) => setFilters((prev) => ({ ...prev, search_term: e.target.value, page: 1 }))}
            placeholder="Search Case #, Account ID, or Merchant Category..."
            className="queue-search-input"
          />
        </div>

        <div className="queue-filter-group">
          <select
            value={filters.status || ''}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                status: (e.target.value as CaseStatus) || undefined,
                page: 1,
              }))
            }
            className="queue-filter-select"
            aria-label="Filter by case status"
          >
            <option value="">All Statuses</option>
            <option value="OPEN">Open</option>
            <option value="IN_REVIEW">In Review</option>
            <option value="ESCALATED">Escalated</option>
            <option value="RESOLVED">Resolved</option>
            <option value="CLOSED">Closed</option>
          </select>

          <select
            value={filters.priority || ''}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                priority: (e.target.value as CasePriority) || undefined,
                page: 1,
              }))
            }
            className="queue-filter-select"
            aria-label="Filter by priority"
          >
            <option value="">All Priorities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>

          <select
            value={filters.assigned_to || ''}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                assigned_to: e.target.value || undefined,
                page: 1,
              }))
            }
            className="queue-filter-select"
            aria-label="Filter by assignment"
          >
            <option value="">All Assignments</option>
            <option value="unassigned">Unassigned Only</option>
            <option value="analyst_01">Assigned: analyst_01</option>
          </select>

          <select
            value={filters.risk_tier || ''}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                risk_tier: (e.target.value as RiskTier) || undefined,
                page: 1,
              }))
            }
            className="queue-filter-select"
            aria-label="Filter by risk tier"
          >
            <option value="">All Risk Tiers</option>
            <option value="LOW">Low Risk</option>
            <option value="MEDIUM">Medium Risk</option>
            <option value="HIGH">High Risk</option>
            <option value="CRITICAL">Critical Risk</option>
          </select>

          <select
            value={filters.limit}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, limit: Number(e.target.value), page: 1 }))
            }
            className="queue-filter-select"
            aria-label="Rows per page"
          >
            <option value={10}>10 per page</option>
            <option value={20}>20 per page</option>
            <option value={50}>50 per page</option>
          </select>
        </div>

        {/* Polling & Refresh Toolbar */}
        <div className="queue-sync-controls">
          <div className="polling-status-chip">
            <span className={`polling-indicator-dot ${isAutoPolling ? 'active' : 'inactive'}`} />
            <span className="polling-status-text">
              {isAutoPolling ? 'Queue Polling' : 'Paused'}
            </span>
          </div>

          <button
            type="button"
            onClick={toggleAutoPolling}
            className={`btn-control ${isAutoPolling ? 'btn-active' : ''}`}
            title={isAutoPolling ? 'Pause queue polling' : 'Resume queue polling'}
          >
            {isAutoPolling ? 'Pause' : 'Resume'}
          </button>

          <button
            type="button"
            onClick={handleManualRefresh}
            disabled={isPolling}
            className="btn-control"
            title="Refresh queue"
          >
            <RefreshCw
              size={13}
              style={{ animation: isPolling ? 'spin 1s linear infinite' : 'none' }}
            />
            <span>Refresh</span>
          </button>
        </div>
      </section>

      {/* Content Area: Table / Loading / Empty */}
      {error && (
        <ErrorBanner
          title="Queue Retrieval Error"
          message={error}
          onRetry={handleManualRefresh}
        />
      )}

      {loading && !data ? (
        <LoadingSpinner message="Querying case database records..." />
      ) : data && data.items.length > 0 ? (
        <div className="queue-table-card">
          <div className="transaction-table-wrapper">
            <table className="transaction-table queue-table">
              <thead>
                <tr>
                  <th>Case #</th>
                  <th
                    className="sortable-header"
                    onClick={() => handleSort('priority')}
                    title="Sort by Priority"
                  >
                    <span>Priority</span>
                    <ArrowUpDown size={12} />
                  </th>
                  <th>Status</th>
                  <th
                    className="sortable-header"
                    onClick={() => handleSort('opened_at')}
                    title="Sort by Opened Timestamp"
                  >
                    <span>Opened At</span>
                    <ArrowUpDown size={12} />
                  </th>
                  <th>Account ID</th>
                  <th>Merchant Category</th>
                  <th>Amount</th>
                  <th
                    className="sortable-header"
                    onClick={() => handleSort('risk_score')}
                    title="Sort by Risk Score"
                  >
                    <span>Risk Score</span>
                    <ArrowUpDown size={12} />
                  </th>
                  <th>Decision</th>
                  <th>Assigned To</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item: CaseQueueItem) => (
                  <tr
                    key={item.id}
                    className="transaction-row clickable-row"
                    onClick={() => onSelectCase(item.id)}
                    tabIndex={0}
                    role="button"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onSelectCase(item.id);
                      }
                    }}
                    aria-label={`Open investigation workspace for case ${item.case_number}`}
                  >
                    <td className="cell-id">
                      <strong className="font-mono text-primary">{item.case_number}</strong>
                    </td>
                    <td>{renderPriorityBadge(item.priority)}</td>
                    <td>{renderStatusBadge(item.status)}</td>
                    <td className="cell-time">
                      <Clock size={11} className="cell-icon-subtle" />
                      <span>{formatDateTime(item.opened_at)}</span>
                    </td>
                    <td className="cell-id font-mono text-muted">{item.account_id || '—'}</td>
                    <td className="cell-category">
                      <span className="category-tag">{item.merchant_category || '—'}</span>
                    </td>
                    <td className="cell-amount">
                      <strong>{formatCurrency(item.amount, item.currency)}</strong>
                    </td>
                    <td className="cell-score">
                      <RiskScoreMeter score={item.risk_score} tier={item.risk_tier} />
                    </td>
                    <td>{renderDecisionBadge(item.decision_action)}</td>
                    <td className="cell-assignee">
                      {item.assigned_to ? (
                        <span className="assignee-badge font-mono">
                          <User size={11} /> {item.assigned_to}
                        </span>
                      ) : (
                        <span className="unassigned-tag">Unassigned</span>
                      )}
                    </td>
                    <td className="cell-actions" onClick={(e) => e.stopPropagation()}>
                      {!item.assigned_to && item.status !== 'RESOLVED' && item.status !== 'CLOSED' && (
                        <button
                          type="button"
                          className="btn-table-claim"
                          onClick={(e) => handleClaim(e, item.id)}
                          title="Claim this case directly"
                        >
                          <UserCheck size={12} />
                          <span>Claim</span>
                        </button>
                      )}
                      <button
                        type="button"
                        className="btn-table-investigate"
                        onClick={() => onSelectCase(item.id)}
                        title="Open full investigation workspace"
                      >
                        Investigate
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination Footer */}
          <div className="queue-pagination-bar">
            <div className="pagination-info">
              Showing {(filters.page - 1) * filters.limit + 1} –{' '}
              {Math.min(filters.page * filters.limit, data.total)} of {data.total} cases
            </div>

            <div className="pagination-controls">
              <button
                type="button"
                className="btn-pagination"
                onClick={() => setFilters((prev) => ({ ...prev, page: Math.max(1, prev.page - 1) }))}
                disabled={filters.page <= 1}
                aria-label="Previous Page"
              >
                <ChevronLeft size={16} />
                <span>Previous</span>
              </button>

              <span className="page-number-indicator">
                Page {filters.page} of {totalPages}
              </span>

              <button
                type="button"
                className="btn-pagination"
                onClick={() => setFilters((prev) => ({ ...prev, page: Math.min(totalPages, prev.page + 1) }))}
                disabled={filters.page >= totalPages}
                aria-label="Next Page"
              >
                <span>Next</span>
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </div>
      ) : (
        <EmptyState
          title="No Cases Match Active Filters"
          description="Adjust your search criteria, triage filters, or clear priority selections to view active review cases."
        />
      )}

      {/* Manual Escalation Modal */}
      {isCreateModalOpen && (
        <CreateCaseModal
          isOpen={isCreateModalOpen}
          onClose={() => setIsCreateModalOpen(false)}
          transactionId={createTxIdInput}
          onSuccess={(created) => {
            handleManualRefresh();
            onSelectCase(created.id);
          }}
          onOpenExistingCase={(existingId) => {
            onSelectCase(existingId);
          }}
        />
      )}
    </div>
  );
};
