import React, { useEffect, useState, useCallback } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Zap,
  Gauge,
  CircleDollarSign,
} from 'lucide-react';
import { Navbar, type DashboardTab } from './components/common/Navbar.tsx';
import { StatCard } from './components/common/StatCard.tsx';
import { Badge } from './components/common/Badge.tsx';
import { LoadingSpinner } from './components/common/LoadingSpinner.tsx';
import { ErrorBanner } from './components/common/ErrorBanner.tsx';
import { EmptyState } from './components/common/EmptyState.tsx';
import { LiveTransactionFeed } from './components/overview/LiveTransactionFeed.tsx';
import { AnalyticsDashboard } from './components/analytics/AnalyticsDashboard.tsx';
import { WhatIfSimulator } from './components/simulator/WhatIfSimulator.tsx';
import { fetchDashboardOverview, fetchHealthStatus } from './api/dashboardApi.ts';
import type {
  DashboardOverviewResponse,
  HealthResponse,
  TransactionDetailResponse,
} from './types/api.ts';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<DashboardTab>('operations');
  const [baselineTransaction, setBaselineTransaction] = useState<TransactionDetailResponse | null>(null);
  const [overview, setOverview] = useState<DashboardOverviewResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  const loadHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const healthData = await fetchHealthStatus();
      setHealth(healthData);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Health service unreachable';
      setHealthError(message);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  const loadOverview = useCallback(async () => {
    setError(null);
    try {
      const overviewData = await fetchDashboardOverview();
      setOverview(overviewData);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch dashboard metrics';
      setError(message);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  const handleRefresh = useCallback(() => {
    setIsRefreshing(true);
    loadHealth();
    loadOverview();
  }, [loadHealth, loadOverview]);

  useEffect(() => {
    loadHealth();
    loadOverview();
  }, [loadHealth, loadOverview]);

  const handleSimulateTransaction = useCallback((detail: TransactionDetailResponse) => {
    setBaselineTransaction(detail);
    setActiveTab('simulator');
  }, []);

  const formatCurrency = (val: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(val);
  };

  return (
    <div className="dashboard-layout">
      <Navbar
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onRefresh={handleRefresh}
        isRefreshing={isRefreshing}
      />

      <main className="dashboard-main">
        {activeTab === 'simulator' ? (
          <WhatIfSimulator
            initialBaselineTransaction={baselineTransaction}
            onClearBaseline={() => setBaselineTransaction(null)}
          />
        ) : activeTab === 'analytics' ? (
          <AnalyticsDashboard />
        ) : (
          <>
            <header className="dashboard-header">
              <div>
                <h1 className="dashboard-headline">Operational Overview</h1>
                <p className="dashboard-subheadline">
                  Real-time telemetry, transaction metrics, and automated decision distribution
                </p>
              </div>
            </header>

            {error && (
              <ErrorBanner
                title="Overview Telemetry Error"
                message={error}
                onRetry={handleRefresh}
              />
            )}

            {loading ? (
              <LoadingSpinner message="Querying database metrics and operational telemetry..." />
            ) : overview ? (
              <>
                <section aria-label="Key Performance Indicators">
                  <h2 className="section-title">
                    <Activity size={18} />
                    <span>Operational KPI Grid</span>
                  </h2>

                  <div className="stats-grid">
                    <StatCard
                      title="Total Transactions"
                      value={overview.total_transactions.toLocaleString()}
                      icon={<Activity size={18} />}
                      sublabel="Total Volume"
                      subvalue={formatCurrency(overview.total_amount)}
                      tone="primary"
                    />

                    <StatCard
                      title="Approved Decisions"
                      value={overview.approval_count.toLocaleString()}
                      icon={<CheckCircle2 size={18} style={{ color: 'var(--status-approve)' }} />}
                      sublabel="Approval Rate"
                      subvalue={`${overview.approval_rate.toFixed(2)}%`}
                      tone="approve"
                      badge={<Badge variant="approve">APPROVE</Badge>}
                    />

                    <StatCard
                      title="Manual Review Queue"
                      value={overview.review_count.toLocaleString()}
                      icon={<AlertTriangle size={18} style={{ color: 'var(--status-review)' }} />}
                      sublabel="Escalation Rate"
                      subvalue={`${overview.review_rate.toFixed(2)}%`}
                      tone="review"
                      badge={<Badge variant="review">REVIEW</Badge>}
                    />

                    <StatCard
                      title="Automated Blocks"
                      value={overview.block_count.toLocaleString()}
                      icon={<ShieldAlert size={18} style={{ color: 'var(--status-block)' }} />}
                      sublabel="Block Rate"
                      subvalue={`${overview.block_rate.toFixed(2)}%`}
                      tone="block"
                      badge={<Badge variant="block">BLOCK</Badge>}
                    />

                    <StatCard
                      title="Mean Risk Score"
                      value={`${overview.average_risk_score.toFixed(1)}`}
                      icon={<Gauge size={18} style={{ color: 'var(--status-info)' }} />}
                      sublabel="Calibrated Score"
                      subvalue="0 – 100 Scale"
                      tone="info"
                      badge={<Badge variant="info">RISK INDEX</Badge>}
                    />

                    <StatCard
                      title="Average Pipeline Latency"
                      value={`${overview.average_latency_ms.toFixed(2)} ms`}
                      icon={<Zap size={18} style={{ color: 'var(--primary)' }} />}
                      sublabel="Target SLA"
                      subvalue="< 50.0 ms"
                      tone="primary"
                      badge={<Badge variant="neutral">SLA PASS</Badge>}
                    />
                  </div>
                </section>

                {overview.total_transactions === 0 && (
                  <EmptyState
                    title="No Transactions Recorded"
                    description="The PostgreSQL database contains 0 evaluated transactions. Incoming transactions scored by the risk engine or generated by benchmark suites will populate this dashboard."
                    icon={<CircleDollarSign size={28} />}
                  />
                )}

                <LiveTransactionFeed
                  onRefreshTriggered={loadOverview}
                  onSimulateTransaction={handleSimulateTransaction}
                />
              </>
            ) : null}
          </>
        )}
      </main>

      <footer className="footer">
        <div>
          AI-Powered Fraud Detection &amp; Risk Intelligence Platform &copy; 2026
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <span className="footer-link">Phase 11.5 What-If Transaction Simulator</span>
          <span>•</span>
          <span className="footer-link">Near-Real-Time Risk Intelligence</span>
        </div>
      </footer>
    </div>
  );
};


export default App;

