import React, { useEffect, useState, useCallback } from 'react';
import {
  Search,
  Filter,
  ArrowUpDown,
  Database,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Eye,
  SlidersHorizontal,
} from 'lucide-react';
import { fetchFeatureDriftList } from '../../api/monitoringApi.ts';
import { FeatureDetailModal } from './FeatureDetailModal.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { EmptyState } from '../common/EmptyState.tsx';
import type {
  FeatureDriftItemResponse,
  FeatureDriftListResponse,
  FeatureSortKey,
  MonitoringTimeWindow,
} from '../../types/monitoring.ts';

interface FeatureDriftTabProps {
  window: MonitoringTimeWindow;
  startTime?: string;
  endTime?: string;
  modelVersion?: string;
}

export const FeatureDriftTab: React.FC<FeatureDriftTabProps> = ({
  window,
  startTime,
  endTime,
  modelVersion = '1.0.0',
}) => {
  const [data, setData] = useState<FeatureDriftListResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters & Sorting state
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [sortBy, setSortBy] = useState<FeatureSortKey>('psi');

  // Modal inspection state
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const loadFeatureDrift = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchFeatureDriftList({
        window,
        start_time: startTime,
        end_time: endTime,
        category: categoryFilter !== 'all' ? categoryFilter : undefined,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        search_term: searchTerm.trim() || undefined,
        sort_by: sortBy,
        limit: 55,
        offset: 0,
        model_version: modelVersion,
      });
      setData(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch feature drift report');
    } finally {
      setLoading(false);
    }
  }, [window, startTime, endTime, categoryFilter, statusFilter, searchTerm, sortBy, modelVersion]);

  useEffect(() => {
    loadFeatureDrift();
  }, [loadFeatureDrift]);

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
      default:
        return <span className="badge badge-neutral">{status}</span>;
    }
  };

  return (
    <div className="monitoring-tab-content">
      {/* 1. Header Summary Stats */}
      {data && (
        <div className="feature-drift-summary-row">
          <div className="f-stat-card">
            <span className="f-lbl">Monitored Features</span>
            <strong className="f-val font-mono">{data.total_features_count}</strong>
            <span className="f-sub">Canonical 55-feature pipeline</span>
          </div>
          <div className="f-stat-card">
            <span className="f-lbl">Drifted Features</span>
            <strong className={`f-val font-mono ${data.drifted_features_count > 0 ? 'text-amber' : 'text-green'}`}>
              {data.drifted_features_count}
            </strong>
            <span className="f-sub">Warning or Critical severity</span>
          </div>
          <div className="f-stat-card">
            <span className="f-lbl">Critical Drift Count</span>
            <strong className={`f-val font-mono ${data.critical_features_count > 0 ? 'text-red' : ''}`}>
              {data.critical_features_count}
            </strong>
            <span className="f-sub">PSI &ge; 0.25 or KS p &le; 0.01</span>
          </div>
          <div className="f-stat-card">
            <span className="f-lbl">Evaluated Window Size</span>
            <strong className="f-val font-mono">{data.sample_count.toLocaleString()}</strong>
            <span className="f-sub">Transactions in window</span>
          </div>
        </div>
      )}

      {/* 2. Interactive Filter & Search Toolbar */}
      <div className="monitoring-filter-toolbar">
        <div className="search-box">
          <Search size={15} className="search-icon" />
          <input
            type="text"
            className="search-input font-mono"
            placeholder="Search feature name (e.g. amt_to_mean_ratio_30d)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <div className="filter-select-wrapper">
            <Filter size={14} className="filter-icon" />
            <select
              className="filter-select"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              aria-label="Filter by feature domain group"
            >
              <option value="all">All Domain Groups</option>
              <option value="amount">Amount &amp; Ratios</option>
              <option value="velocity">Velocity &amp; Frequency</option>
              <option value="geographic">Geographic &amp; Distance</option>
              <option value="demographic">Demographic &amp; Job</option>
              <option value="behavioral">Behavioral Patterns</option>
            </select>
          </div>

          <div className="filter-select-wrapper">
            <SlidersHorizontal size={14} className="filter-icon" />
            <select
              className="filter-select"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              aria-label="Filter by drift status"
            >
              <option value="all">All Statuses</option>
              <option value="CRITICAL">Critical Drift</option>
              <option value="WARNING">Warning Drift</option>
              <option value="NORMAL">Normal / Nominal</option>
            </select>
          </div>

          <div className="filter-select-wrapper">
            <ArrowUpDown size={14} className="filter-icon" />
            <select
              className="filter-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as FeatureSortKey)}
              aria-label="Sort features by metric"
            >
              <option value="psi">Sort by PSI (Descending)</option>
              <option value="ks_stat">Sort by KS Statistic</option>
              <option value="missing_delta">Sort by Missing Rate Delta</option>
            </select>
          </div>
        </div>
      </div>

      {/* 3. Main 55-Feature Table or States */}
      {error && (
        <ErrorBanner
          title="Feature Drift Telemetry Error"
          message={error}
          onRetry={loadFeatureDrift}
        />
      )}

      {loading ? (
        <LoadingSpinner message="Calculating feature-level drift statistics across 55 canonical features..." />
      ) : data && data.items.length === 0 ? (
        <EmptyState
          title="No Matching Features Found"
          description="No canonical features match the applied category, status, or search filters."
          icon={<Database size={28} />}
        />
      ) : data ? (
        <div className="feature-drift-table-container">
          <table className="monitoring-table feature-table">
            <thead>
              <tr>
                <th>Feature Name</th>
                <th>Category</th>
                <th>Type</th>
                <th>Status</th>
                <th>PSI Value</th>
                <th>Two-Sample KS / JSD</th>
                <th>Missing Rate &amp; Delta</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((feat: FeatureDriftItemResponse) => {
                const psiPct = Math.min(100, (feat.psi / 0.3) * 100);
                const psiColor =
                  feat.status === 'CRITICAL'
                    ? 'var(--status-block)'
                    : feat.status === 'WARNING'
                    ? 'var(--status-review)'
                    : 'var(--status-approve)';

                return (
                  <tr key={feat.feature_name} className={`row-status-${feat.status.toLowerCase()}`}>
                    <td>
                      <strong className="font-mono feature-name-cell">{feat.feature_name}</strong>
                    </td>
                    <td>
                      <span className="category-pill">{feat.category}</span>
                    </td>
                    <td>
                      <span className="type-pill font-mono">{feat.feature_type.slice(0, 3).toUpperCase()}</span>
                    </td>
                    <td>{getStatusBadge(feat.status)}</td>
                    <td>
                      <div className="metric-meter-cell">
                        <div className="meter-track">
                          <div
                            className="meter-fill"
                            style={{ width: `${psiPct}%`, backgroundColor: psiColor }}
                          />
                        </div>
                        <span className="font-mono val-text">{feat.psi.toFixed(4)}</span>
                      </div>
                    </td>
                    <td>
                      {feat.feature_type === 'numerical' ? (
                        <div className="ks-metric-cell">
                          <span className="font-mono">
                            KS: {feat.ks_statistic !== null ? feat.ks_statistic.toFixed(3) : 'N/A'}
                          </span>
                          <span className="font-mono text-muted text-xs">
                            p:{' '}
                            {feat.ks_p_value !== null
                              ? feat.ks_p_value < 0.001
                                ? feat.ks_p_value.toExponential(1)
                                : feat.ks_p_value.toFixed(3)
                              : 'N/A'}
                          </span>
                        </div>
                      ) : (
                        <span className="font-mono">
                          JSD: {feat.js_divergence !== null ? feat.js_divergence.toFixed(4) : 'N/A'}
                        </span>
                      )}
                    </td>
                    <td>
                      <div className="missing-rate-cell">
                        <span className="font-mono">
                          {(feat.missing_rate_current * 100).toFixed(1)}%
                        </span>
                        {Math.abs(feat.missing_rate_delta) > 0.001 && (
                          <span
                            className={`font-mono text-xs ${
                              Math.abs(feat.missing_rate_delta) > 0.05 ? 'text-amber' : 'text-muted'
                            }`}
                          >
                            (&Delta;{(feat.missing_rate_delta * 100).toFixed(1)}%)
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn-inspect-feature"
                        onClick={() => setSelectedFeature(feat.feature_name)}
                        title={`Inspect ${feat.feature_name} distributions`}
                      >
                        <Eye size={14} />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      {/* 4. Single-Feature Detail Modal */}
      {selectedFeature && (
        <FeatureDetailModal
          featureName={selectedFeature}
          timeWindow={window}
          startTime={startTime}
          endTime={endTime}
          modelVersion={modelVersion}
          onClose={() => setSelectedFeature(null)}
        />
      )}
    </div>
  );
};
