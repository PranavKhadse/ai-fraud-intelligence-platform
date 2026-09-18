import React, { useState, useCallback } from 'react';
import {
  Clock,
  RefreshCw,
  SlidersHorizontal,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Zap,
  ArrowRightLeft,
  Calendar,
} from 'lucide-react';
import { Badge } from '../common/Badge.tsx';
import { RiskScoreMeter } from '../common/RiskScoreMeter.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { EmptyState } from '../common/EmptyState.tsx';
import { TransactionDrawer } from '../investigation/TransactionDrawer.tsx';
import { usePolling } from '../../hooks/usePolling.ts';
import { fetchTransactions } from '../../api/dashboardApi.ts';
import type {
  DecisionAction,
  RiskTier,
  TransactionListItem,
  TransactionListResponse,
  TransactionQueryParams,
  TransactionDetailResponse,
} from '../../types/api.ts';

interface LiveTransactionFeedProps {
  onRefreshTriggered?: () => void;
  onSimulateTransaction?: (detail: TransactionDetailResponse) => void;
  onOpenCase?: (caseId: string) => void;
}

export const LiveTransactionFeed: React.FC<LiveTransactionFeedProps> = ({
  onRefreshTriggered,
  onSimulateTransaction,
  onOpenCase,
}) => {

  const [selectedTransactionId, setSelectedTransactionId] = useState<string | null>(null);
  const [decisionFilter, setDecisionFilter] = useState<DecisionAction | undefined>(undefined);
  const [tierFilter, setTierFilter] = useState<RiskTier | undefined>(undefined);
  const [limit, setLimit] = useState<number>(20);

  const queryParams: TransactionQueryParams = {
    limit,
    offset: 0,
    decision_action: decisionFilter,
    risk_tier: tierFilter,
  };

  const fetchFn = useCallback(() => {
    return fetchTransactions(queryParams);
  }, [limit, decisionFilter, tierFilter]);

  const {
    data,
    loading,
    isPolling,
    error,
    lastUpdated,
    isAutoPolling,
    interval,
    setInterval,
    toggleAutoPolling,
    refresh,
  } = usePolling<TransactionListResponse>(fetchFn, {
    interval: 5000,
    enabled: true,
  });

  const handleManualRefresh = async () => {
    await refresh();
    if (onRefreshTriggered) {
      onRefreshTriggered();
    }
  };

  const formatDateTime = (isoString: string): string => {
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  const formatCurrency = (val: number, currency: string): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency || 'USD',
      minimumFractionDigits: 2,
    }).format(val);
  };

  const renderDecisionBadge = (action: DecisionAction) => {
    switch (action) {
      case 'APPROVE':
        return (
          <Badge variant="approve" icon={<CheckCircle2 size={12} />}>
            APPROVE
          </Badge>
        );
      case 'REVIEW':
        return (
          <Badge variant="review" icon={<AlertTriangle size={12} />}>
            REVIEW
          </Badge>
        );
      case 'BLOCK':
        return (
          <Badge variant="block" icon={<ShieldAlert size={12} />}>
            BLOCK
          </Badge>
        );
    }
  };

  const renderTierBadge = (tier: RiskTier) => {
    switch (tier) {
      case 'LOW':
        return <Badge variant="approve">LOW</Badge>;
      case 'MEDIUM':
        return <Badge variant="review">MED</Badge>;
      case 'HIGH':
        return <Badge variant="block">HIGH</Badge>;
      case 'CRITICAL':
        return <Badge variant="block">CRIT</Badge>;
    }
  };

  return (
    <section className="live-feed-section" aria-label="Real-Time Risk Intelligence Feed">
      <div className="live-feed-header">
        <div className="live-feed-title-wrap">
          <div className="live-feed-title">
            <Zap size={20} className="pulse-icon" />
            <h3>Live Transaction Feed</h3>
          </div>
          <p className="live-feed-description">
            Near-real-time ingestion stream of evaluated financial transactions
          </p>
        </div>

        {/* Polling & Feed Controls Toolbar */}
        <div className="feed-controls-toolbar">
          <div className="polling-status-chip">
            <span
              className={`polling-indicator-dot ${isAutoPolling ? 'active' : 'inactive'}`}
            />
            <span className="polling-status-text">
              {isAutoPolling ? 'Live Polling' : 'Paused'}
            </span>
          </div>

          <div className="control-group">
            <label htmlFor="poll-interval-select" className="control-label">
              Interval:
            </label>
            <select
              id="poll-interval-select"
              value={interval}
              onChange={(e) => setInterval(Number(e.target.value))}
              className="control-select"
              disabled={!isAutoPolling}
            >
              <option value={5000}>5s</option>
              <option value={10000}>10s</option>
              <option value={30000}>30s</option>
            </select>
          </div>

          <button
            type="button"
            onClick={toggleAutoPolling}
            className={`btn-control ${isAutoPolling ? 'btn-active' : ''}`}
            title={isAutoPolling ? 'Pause live polling' : 'Resume live polling'}
          >
            {isAutoPolling ? 'Pause' : 'Resume'}
          </button>

          <button
            type="button"
            onClick={handleManualRefresh}
            disabled={isPolling}
            className="btn-control"
            title="Force immediate refresh"
          >
            <RefreshCw
              size={13}
              style={{
                animation: isPolling ? 'spin 1s linear infinite' : 'none',
              }}
            />
            <span>Refresh</span>
          </button>

          {lastUpdated && (
            <div className="last-updated-text" title={`Last synced at ${lastUpdated.toISOString()}`}>
              <Clock size={12} />
              <span>{formatDateTime(lastUpdated.toISOString())}</span>
            </div>
          )}
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="feed-filter-bar">
        <div className="filter-group">
          <span className="filter-label">
            <SlidersHorizontal size={14} />
            <span>Filters:</span>
          </span>

          <select
            value={decisionFilter || ''}
            onChange={(e) => setDecisionFilter((e.target.value as DecisionAction) || undefined)}
            className="filter-select"
            aria-label="Filter by decision"
          >
            <option value="">All Decisions</option>
            <option value="APPROVE">Approve</option>
            <option value="REVIEW">Review</option>
            <option value="BLOCK">Block</option>
          </select>

          <select
            value={tierFilter || ''}
            onChange={(e) => setTierFilter((e.target.value as RiskTier) || undefined)}
            className="filter-select"
            aria-label="Filter by risk tier"
          >
            <option value="">All Risk Tiers</option>
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
            <option value="CRITICAL">Critical</option>
          </select>

          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="filter-select"
            aria-label="Items per page"
          >
            <option value={10}>Show 10</option>
            <option value={20}>Show 20</option>
            <option value={50}>Show 50</option>
          </select>
        </div>

        {data && (
          <div className="feed-total-count">
            <span>Showing {data.items.length} of {data.total_count} transactions</span>
          </div>
        )}
      </div>

      {/* Content Area */}
      {error && (
        <ErrorBanner
          title="Transaction Feed Error"
          message={error}
          onRetry={handleManualRefresh}
        />
      )}

      {loading && !data ? (
        <LoadingSpinner message="Streaming live transaction events..." />
      ) : data && data.items.length > 0 ? (
        <div className="transaction-table-wrapper">
          <table className="transaction-table">
            <thead>
              <tr>
                <th>Time (UTC)</th>
                <th>Transaction ID</th>
                <th>Category</th>
                <th>Amount</th>
                <th>Risk Score</th>
                <th>Risk Tier</th>
                <th>Decision</th>
                <th>Rules Override</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((item: TransactionListItem) => (
                <tr
                  key={item.id}
                  className={`transaction-row clickable-row ${
                    selectedTransactionId === item.id ? 'row-selected' : ''
                  }`}
                  onClick={() => setSelectedTransactionId(item.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelectedTransactionId(item.id);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`Investigate transaction ${item.external_transaction_id || item.id}`}
                  title="Click to open deep investigation drawer"
                >
                  <td className="cell-time">
                    <Calendar size={12} className="cell-icon-subtle" />
                    <span>{formatDateTime(item.transaction_timestamp)}</span>
                  </td>
                  <td className="cell-id">
                    <span className="font-mono">{item.external_transaction_id || item.id.substring(0, 13) + '...'}</span>
                  </td>
                  <td className="cell-category">
                    <span className="category-tag">{item.merchant_category}</span>
                  </td>
                  <td className="cell-amount">
                    <strong>{formatCurrency(item.amount, item.currency)}</strong>
                  </td>
                  <td className="cell-score">
                    <RiskScoreMeter score={item.risk_score} tier={item.risk_tier} />
                  </td>
                  <td className="cell-tier">{renderTierBadge(item.risk_tier)}</td>
                  <td className="cell-decision">{renderDecisionBadge(item.decision_action)}</td>
                  <td className="cell-override">
                    {item.is_overridden ? (
                      <Badge variant="review" icon={<ArrowRightLeft size={11} />}>
                        OVERRIDDEN
                      </Badge>
                    ) : (
                      <span className="text-muted" style={{ fontSize: '0.75rem' }}>
                        Baseline
                      </span>
                    )}
                  </td>
                  <td className="cell-latency">
                    {item.evaluation_latency_ms !== null ? (
                      <span className="font-mono text-muted">{item.evaluation_latency_ms.toFixed(1)} ms</span>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          title="No Matching Transactions"
          description="No transaction records match the selected filter criteria or the database has not processed transactions yet."
        />
      )}

      {/* Transaction Deep Investigation Drawer */}
      <TransactionDrawer
        transactionId={selectedTransactionId}
        onClose={() => setSelectedTransactionId(null)}
        onSimulateTransaction={onSimulateTransaction}
        onOpenCase={onOpenCase}
      />
    </section>
  );
};

