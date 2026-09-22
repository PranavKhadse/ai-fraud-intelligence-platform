import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  Gauge,
  Activity,
  Database,
  TrendingUp,
  BarChart3,
  Layers,
  RefreshCw,
  Clock,
} from 'lucide-react';
import { fetchMonitoringHealth } from '../../api/monitoringApi.ts';
import { MonitoringOverviewTab } from './MonitoringOverviewTab.tsx';
import { FeatureDriftTab } from './FeatureDriftTab.tsx';
import { PredictionDriftTab } from './PredictionDriftTab.tsx';
import { ModelPerformanceTab } from './ModelPerformanceTab.tsx';
import { MonitoringSnapshotsTab } from './MonitoringSnapshotsTab.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import type {
  MonitoringHealthResponse,
  MonitoringTimeWindow,
} from '../../types/monitoring.ts';

export type MonitoringSubTab = 'overview' | 'features' | 'predictions' | 'performance' | 'snapshots';

interface MonitoringDashboardProps {
  onAlertCountChange?: (count: number) => void;
}

export const MonitoringDashboard: React.FC<MonitoringDashboardProps> = ({
  onAlertCountChange,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<MonitoringSubTab>('overview');
  const [timeWindow, setTimeWindow] = useState<MonitoringTimeWindow>('24h');
  const [customStart, setCustomStart] = useState<string>('');
  const [customEnd, setCustomEnd] = useState<string>('');
  const [modelVersion] = useState<string>('1.0.0');

  // Polling Auto-Refresh state (Explicit REST polling only, no WebSockets)
  const [pollingInterval, setPollingInterval] = useState<number>(0); // 0 = Off, 30 = 30s, 60 = 60s
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // High-level health state
  const [healthData, setHealthData] = useState<MonitoringHealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Subtab refresh trigger token
  const [refreshKey, setRefreshKey] = useState<number>(0);

  const onAlertCountChangeRef = useRef(onAlertCountChange);
  useEffect(() => {
    onAlertCountChangeRef.current = onAlertCountChange;
  }, [onAlertCountChange]);

  const loadHealth = useCallback(async (quiet: boolean = false) => {
    if (!quiet) setLoading(true);
    setError(null);
    try {
      const resp = await fetchMonitoringHealth({
        window: timeWindow,
        start_time: timeWindow === 'custom' && customStart ? new Date(customStart).toISOString() : undefined,
        end_time: timeWindow === 'custom' && customEnd ? new Date(customEnd).toISOString() : undefined,
        model_version: modelVersion,
      });
      setHealthData(resp);
      if (onAlertCountChangeRef.current) {
        onAlertCountChangeRef.current(resp.active_alert_count);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to connect to Monitoring Service';
      setError(msg);
    } finally {
      if (!quiet) setLoading(false);
      setIsRefreshing(false);
    }
  }, [timeWindow, customStart, customEnd, modelVersion]);

  // Initial load and window change
  useEffect(() => {
    loadHealth();
  }, [loadHealth, refreshKey]);

  // Polling timer setup
  useEffect(() => {
    if (pollingInterval <= 0) return;

    const timer = setInterval(() => {
      loadHealth(true);
      setRefreshKey((k) => k + 1);
    }, pollingInterval * 1000);

    return () => clearInterval(timer);
  }, [pollingInterval, loadHealth]);

  const handleManualRefresh = () => {
    setIsRefreshing(true);
    loadHealth();
    setRefreshKey((k) => k + 1);
  };

  const isoStart = timeWindow === 'custom' && customStart ? new Date(customStart).toISOString() : undefined;
  const isoEnd = timeWindow === 'custom' && customEnd ? new Date(customEnd).toISOString() : undefined;

  return (
    <div className="monitoring-dashboard">
      {/* Dashboard Top Header & Control Bar */}
      <header className="monitoring-header">
        <div className="monitoring-header-titles">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
            <div className="monitoring-icon-badge">
              <Gauge size={22} />
            </div>
            <div>
              <h1 className="dashboard-headline">ML &amp; Model Monitoring Intelligence</h1>
              <p className="dashboard-subheadline">
                Near-real-time drift diagnostics, 55-feature empirical profiling, normalized score shifts &amp; ground-truth performance
              </p>
            </div>
          </div>
        </div>

        {/* Global Controls: Time Window, Model Version, Polling & Refresh */}
        <div className="monitoring-controls-bar">
          {/* Time Window Buttons */}
          <div className="time-window-pill-group" role="group" aria-label="Time Window Selector">
            <button
              type="button"
              className={`window-pill ${timeWindow === '1h' ? 'active' : ''}`}
              onClick={() => setTimeWindow('1h')}
            >
              1h (Hourly)
            </button>
            <button
              type="button"
              className={`window-pill ${timeWindow === '24h' ? 'active' : ''}`}
              onClick={() => setTimeWindow('24h')}
            >
              24h (Daily)
            </button>
            <button
              type="button"
              className={`window-pill ${timeWindow === '7d' ? 'active' : ''}`}
              onClick={() => setTimeWindow('7d')}
            >
              7 Days
            </button>
            <button
              type="button"
              className={`window-pill ${timeWindow === '30d' ? 'active' : ''}`}
              onClick={() => setTimeWindow('30d')}
            >
              30 Days
            </button>
            <button
              type="button"
              className={`window-pill ${timeWindow === 'custom' ? 'active' : ''}`}
              onClick={() => setTimeWindow('custom')}
            >
              Custom
            </button>
          </div>

          {/* Custom Date Pickers if custom selected */}
          {timeWindow === 'custom' && (
            <div className="custom-datetime-inputs">
              <div className="datetime-field">
                <span className="lbl">From:</span>
                <input
                  type="datetime-local"
                  className="dt-input font-mono"
                  value={customStart}
                  onChange={(e) => setCustomStart(e.target.value)}
                  aria-label="Custom Start DateTime"
                />
              </div>
              <div className="datetime-field">
                <span className="lbl">To:</span>
                <input
                  type="datetime-local"
                  className="dt-input font-mono"
                  value={customEnd}
                  onChange={(e) => setCustomEnd(e.target.value)}
                  aria-label="Custom End DateTime"
                />
              </div>
            </div>
          )}

          {/* Model Version Tag */}
          <div className="model-version-selector font-mono" title="Evaluated Champion Model Release">
            <span>Model: v{modelVersion}</span>
          </div>

          {/* Polling Interval Selector (Near-Real-Time REST Polling) */}
          <div className="polling-selector-wrapper">
            <Clock size={14} className="text-muted" />
            <select
              className="polling-select"
              value={pollingInterval}
              onChange={(e) => setPollingInterval(Number(e.target.value))}
              aria-label="Auto-refresh polling interval"
            >
              <option value={0}>Auto-Poll: Off</option>
              <option value={30}>Auto-Poll: 30s</option>
              <option value={60}>Auto-Poll: 60s</option>
            </select>
          </div>

          {/* Manual Refresh Button */}
          <button
            type="button"
            className="btn-refresh-monitoring"
            onClick={handleManualRefresh}
            disabled={isRefreshing}
            title="Trigger near-real-time refresh of monitoring metrics"
          >
            <RefreshCw
              size={14}
              style={{ animation: isRefreshing ? 'spin 1s linear infinite' : 'none' }}
            />
            <span>Refresh</span>
          </button>
        </div>
      </header>

      {/* Sub-Navigation Tabs */}
      <nav className="monitoring-subnav-tabs" aria-label="Monitoring Subsystem Tabs" role="tablist">
        <button
          type="button"
          className={`monitoring-subnav-btn ${activeSubTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('overview')}
          role="tab"
          aria-selected={activeSubTab === 'overview'}
        >
          <Activity size={16} />
          <span>Health &amp; Active Alerts</span>
          {healthData && healthData.active_alert_count > 0 && (
            <span className="subtab-alert-badge font-mono">{healthData.active_alert_count}</span>
          )}
        </button>

        <button
          type="button"
          className={`monitoring-subnav-btn ${activeSubTab === 'features' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('features')}
          role="tab"
          aria-selected={activeSubTab === 'features'}
        >
          <Database size={16} />
          <span>55-Feature Data Drift</span>
          {healthData && healthData.data_drift_status === 'CRITICAL' && (
            <span className="subtab-dot dot-critical" />
          )}
        </button>

        <button
          type="button"
          className={`monitoring-subnav-btn ${activeSubTab === 'predictions' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('predictions')}
          role="tab"
          aria-selected={activeSubTab === 'predictions'}
        >
          <TrendingUp size={16} />
          <span>Prediction &amp; Score Drift</span>
          {healthData && healthData.prediction_drift_status === 'CRITICAL' && (
            <span className="subtab-dot dot-critical" />
          )}
        </button>

        <button
          type="button"
          className={`monitoring-subnav-btn ${activeSubTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('performance')}
          role="tab"
          aria-selected={activeSubTab === 'performance'}
        >
          <BarChart3 size={16} />
          <span>Ground-Truth Performance</span>
          {healthData && healthData.performance_status === 'CRITICAL' && (
            <span className="subtab-dot dot-critical" />
          )}
        </button>

        <button
          type="button"
          className={`monitoring-subnav-btn ${activeSubTab === 'snapshots' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('snapshots')}
          role="tab"
          aria-selected={activeSubTab === 'snapshots'}
        >
          <Layers size={16} />
          <span>Historical Snapshots</span>
        </button>
      </nav>

      {/* Main Monitoring Body */}
      <main className="monitoring-main-view">
        {error && (
          <ErrorBanner
            title="Monitoring Telemetry Error"
            message={error}
            onRetry={handleManualRefresh}
          />
        )}

        {loading ? (
          <LoadingSpinner message="Querying monitoring telemetry from PostgreSQL and evaluation engines..." />
        ) : activeSubTab === 'overview' ? (
          healthData ? (
            <MonitoringOverviewTab
              health={healthData}
              onNavigateTab={(tab) => setActiveSubTab(tab)}
            />
          ) : null
        ) : activeSubTab === 'features' ? (
          <FeatureDriftTab
            key={`features-${refreshKey}`}
            window={timeWindow}
            startTime={isoStart}
            endTime={isoEnd}
            modelVersion={modelVersion}
          />
        ) : activeSubTab === 'predictions' ? (
          <PredictionDriftTab
            key={`predictions-${refreshKey}`}
            window={timeWindow}
            startTime={isoStart}
            endTime={isoEnd}
            modelVersion={modelVersion}
          />
        ) : activeSubTab === 'performance' ? (
          <ModelPerformanceTab
            key={`performance-${refreshKey}`}
            window={timeWindow}
            startTime={isoStart}
            endTime={isoEnd}
            modelVersion={modelVersion}
          />
        ) : activeSubTab === 'snapshots' ? (
          <MonitoringSnapshotsTab
            key={`snapshots-${refreshKey}`}
            modelVersion={modelVersion}
          />
        ) : null}
      </main>
    </div>
  );
};
