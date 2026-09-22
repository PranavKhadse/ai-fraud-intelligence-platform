import React, { useEffect, useState, useCallback } from 'react';
import {
  Layers,
  Filter,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Activity,
  Calendar,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { fetchMonitoringSnapshots } from '../../api/monitoringApi.ts';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { EmptyState } from '../common/EmptyState.tsx';
import type {
  MonitoringSnapshotItemResponse,
  MonitoringSnapshotListResponse,
} from '../../types/monitoring.ts';

interface MonitoringSnapshotsTabProps {
  modelVersion?: string;
}

export const MonitoringSnapshotsTab: React.FC<MonitoringSnapshotsTabProps> = ({
  modelVersion = '1.0.0',
}) => {
  const [data, setData] = useState<MonitoringSnapshotListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters & Pagination
  const [windowTypeFilter, setWindowTypeFilter] = useState<string>('ALL');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [page, setPage] = useState<number>(1);
  const limit = 20;

  const loadSnapshots = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchMonitoringSnapshots({
        model_version: modelVersion,
        window_type: windowTypeFilter !== 'ALL' ? windowTypeFilter : undefined,
        status: statusFilter !== 'ALL' ? statusFilter : undefined,
        limit,
        offset: (page - 1) * limit,
      });
      setData(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch historical snapshots');
    } finally {
      setLoading(false);
    }
  }, [modelVersion, windowTypeFilter, statusFilter, page, limit]);

  useEffect(() => {
    loadSnapshots();
  }, [loadSnapshots]);

  const getStatusBadge = (status: string) => {
    switch (status.toUpperCase()) {
      case 'CRITICAL':
        return (
          <span className="badge badge-block">
            <ShieldAlert size={13} />
            <span>CRITICAL</span>
          </span>
        );
      case 'WARNING':
        return (
          <span className="badge badge-review">
            <AlertTriangle size={13} />
            <span>WARNING</span>
          </span>
        );
      case 'NORMAL':
        return (
          <span className="badge badge-approve">
            <CheckCircle2 size={13} />
            <span>NORMAL</span>
          </span>
        );
      case 'INSUFFICIENT_DATA':
        return (
          <span className="badge badge-neutral">
            <Activity size={13} />
            <span>INSUFFICIENT</span>
          </span>
        );
      default:
        return <span className="badge badge-neutral">{status}</span>;
    }
  };

  const formatDateRange = (startIso: string, endIso: string) => {
    try {
      const s = new Date(startIso);
      const e = new Date(endIso);
      return `${s.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })} → ${e.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}`;
    } catch {
      return `${startIso} → ${endIso}`;
    }
  };

  const totalPages = data ? Math.ceil(data.total_count / limit) : 1;

  return (
    <div className="monitoring-tab-content">
      {/* 1. Header Toolbar */}
      <div className="monitoring-filter-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={18} className="text-primary" />
          <h3 className="section-title-sm">Persisted Monitoring Snapshots &amp; Rollups</h3>
        </div>

        <div className="filter-group">
          <div className="filter-select-wrapper">
            <Calendar size={14} className="filter-icon" />
            <select
              className="filter-select"
              value={windowTypeFilter}
              onChange={(e) => {
                setWindowTypeFilter(e.target.value);
                setPage(1);
              }}
              aria-label="Filter by window type"
            >
              <option value="ALL">All Window Types</option>
              <option value="HOURLY">HOURLY Rollups (30d retention)</option>
              <option value="DAILY">DAILY Rollups (365d retention)</option>
            </select>
          </div>

          <div className="filter-select-wrapper">
            <Filter size={14} className="filter-icon" />
            <select
              className="filter-select"
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                setPage(1);
              }}
              aria-label="Filter by overall status"
            >
              <option value="ALL">All Statuses</option>
              <option value="CRITICAL">CRITICAL</option>
              <option value="WARNING">WARNING</option>
              <option value="NORMAL">NORMAL</option>
              <option value="INSUFFICIENT_DATA">INSUFFICIENT_DATA</option>
            </select>
          </div>
        </div>
      </div>

      {/* 2. Error and Loading States */}
      {error && (
        <ErrorBanner
          title="Snapshot Telemetry Error"
          message={error}
          onRetry={loadSnapshots}
        />
      )}

      {loading ? (
        <LoadingSpinner message="Querying persisted historical monitoring snapshots from PostgreSQL..." />
      ) : data && data.items.length === 0 ? (
        <EmptyState
          title="No Persisted Monitoring Snapshots Found"
          description="Snapshots are generated automatically by scheduled CLI maintenance tasks or on-demand snapshot computations."
          icon={<Layers size={28} />}
        />
      ) : data ? (
        <>
          <div className="snapshots-table-container">
            <table className="monitoring-table">
              <thead>
                <tr>
                  <th>Snapshot ID</th>
                  <th>Window Type</th>
                  <th>Observation Window</th>
                  <th>Evaluations</th>
                  <th>Labeled Cases</th>
                  <th>Overall Status</th>
                  <th>Data Drift</th>
                  <th>Prediction Drift</th>
                  <th>Performance</th>
                  <th>Generated</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((snap: MonitoringSnapshotItemResponse) => (
                  <tr key={snap.id}>
                    <td>
                      <span className="font-mono text-xs text-muted" title={snap.id}>
                        {snap.id.slice(0, 8)}...
                      </span>
                    </td>
                    <td>
                      <span className="type-pill font-mono">{snap.window_type}</span>
                    </td>
                    <td>
                      <span className="font-mono text-xs">
                        {formatDateRange(snap.window_start, snap.window_end)}
                      </span>
                    </td>
                    <td>
                      <strong className="font-mono">{snap.sample_count.toLocaleString()}</strong>
                    </td>
                    <td>
                      <span className="font-mono">{snap.labeled_count.toLocaleString()}</span>
                    </td>
                    <td>{getStatusBadge(snap.overall_status)}</td>
                    <td>{getStatusBadge(snap.data_drift_status)}</td>
                    <td>{getStatusBadge(snap.prediction_drift_status)}</td>
                    <td>{getStatusBadge(snap.performance_status)}</td>
                    <td>
                      <span className="font-mono text-xs text-muted">
                        {new Date(snap.created_at).toLocaleTimeString([], {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="pagination-bar" style={{ marginTop: '1rem' }}>
              <span className="page-info font-mono text-xs">
                Page {page} of {totalPages} ({data.total_count} total snapshots)
              </span>
              <div className="page-buttons">
                <button
                  type="button"
                  className="btn-page"
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  aria-label="Previous page"
                >
                  <ChevronLeft size={16} />
                  <span>Prev</span>
                </button>
                <button
                  type="button"
                  className="btn-page"
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  aria-label="Next page"
                >
                  <span>Next</span>
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </>
      ) : null}
    </div>
  );
};
