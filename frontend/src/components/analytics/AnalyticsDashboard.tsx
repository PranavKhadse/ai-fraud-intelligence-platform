import React, { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  TrendingUp,
  AlertTriangle,
} from 'lucide-react';
import { TrendCharts } from './TrendCharts.tsx';
import { DistributionCharts } from './DistributionCharts.tsx';
import { RuleAnalyticsTable } from './RuleAnalyticsTable.tsx';
import { LoadingSpinner } from '../common/LoadingSpinner.tsx';
import { ErrorBanner } from '../common/ErrorBanner.tsx';
import { EmptyState } from '../common/EmptyState.tsx';
import {
  fetchAnalyticsDistributions,
  fetchAnalyticsTrends,
  fetchAnalyticsRules,
} from '../../api/dashboardApi.ts';
import type {
  AnalyticsDistributionsResponse,
  AnalyticsRulesResponse,
  AnalyticsTrendsResponse,
} from '../../types/api.ts';

type TimeRangeOption = '24h' | '7d' | '30d' | 'all';

export const AnalyticsDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState<TimeRangeOption>('7d');
  const [distributions, setDistributions] = useState<AnalyticsDistributionsResponse | null>(null);
  const [trends, setTrends] = useState<AnalyticsTrendsResponse | null>(null);
  const [rules, setRules] = useState<AnalyticsRulesResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // Compute ISO time range bounds
  const getBounds = useCallback((range: TimeRangeOption) => {
    const end = new Date();
    let start: Date | undefined;

    switch (range) {
      case '24h':
        start = new Date(end.getTime() - 24 * 60 * 60 * 1000);
        break;
      case '7d':
        start = new Date(end.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case '30d':
        start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
      case 'all':
        start = undefined;
        break;
    }

    return {
      startDate: start ? start.toISOString() : undefined,
      endDate: end.toISOString(),
    };
  }, []);

  const loadAnalytics = useCallback(async (range: TimeRangeOption) => {
    setLoading(true);
    setError(null);

    const { startDate, endDate } = getBounds(range);
    const interval = range === '24h' ? 'hourly' : 'daily';

    try {
      const [distRes, trendRes, rulesRes] = await Promise.all([
        fetchAnalyticsDistributions({ start_date: startDate, end_date: endDate }),
        fetchAnalyticsTrends({ start_date: startDate, end_date: endDate, interval }),
        fetchAnalyticsRules({ start_date: startDate, end_date: endDate, limit: 20 }),
      ]);

      setDistributions(distRes);
      setTrends(trendRes);
      setRules(rulesRes);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to retrieve analytics metrics.';
      setError(msg);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, [getBounds]);

  useEffect(() => {
    loadAnalytics(timeRange);
  }, [timeRange, loadAnalytics]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    loadAnalytics(timeRange);
  };

  return (
    <div className="analytics-dashboard-layout">
      {/* Analytics Sub-Header & Time Range Controls */}
      <div className="analytics-toolbar">
        <div>
          <h2 className="analytics-page-title">
            <TrendingUp size={20} />
            <span>Fraud Analytics &amp; Explainability Telemetry</span>
          </h2>
          <p className="analytics-page-subtitle">
            Longitudinal risk distributions, time-series volume trends, and rule override telemetry
          </p>
        </div>

        <div className="analytics-toolbar-actions">
          <div className="time-range-pills" role="radiogroup" aria-label="Time range selector">
            {(
              [
                { key: '24h', label: 'Past 24h' },
                { key: '7d', label: 'Past 7d' },
                { key: '30d', label: 'Past 30d' },
                { key: 'all', label: 'All Time' },
              ] as const
            ).map(({ key, label }) => (
              <button
                key={key}
                type="button"
                className={`range-pill ${timeRange === key ? 'active' : ''}`}
                onClick={() => setTimeRange(key)}
                aria-checked={timeRange === key}
                role="radio"
              >
                {label}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="analytics-refresh-btn"
            onClick={handleRefresh}
            disabled={isRefreshing || loading}
            aria-label="Refresh analytics data"
          >
            <RefreshCw size={14} className={isRefreshing ? 'spin' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <ErrorBanner
          title="Analytics Retrieval Failed"
          message={error}
          onRetry={() => loadAnalytics(timeRange)}
        />
      )}

      {loading ? (
        <LoadingSpinner message="Aggregating PostgreSQL telemetry, histograms, and time-series trends..." />
      ) : distributions && trends && rules ? (
        <div className="analytics-sections-container">
          {/* Section 1: Time-Series Trends */}
          <section aria-label="Risk and Volume Trends">
            <TrendCharts
              dataPoints={trends.data_points}
              interval={trends.interval}
              startDate={trends.start_date}
              endDate={trends.end_date}
            />
          </section>

          {/* Section 2: Histograms & Categorical Distributions */}
          <section aria-label="Risk Score and Decision Distributions">
            <DistributionCharts
              totalEvaluated={distributions.total_evaluated}
              riskScoreDistribution={distributions.risk_score_distribution}
              modelScoreDistribution={distributions.model_score_distribution}
              riskTierDistribution={distributions.risk_tier_distribution}
              decisionDistribution={distributions.decision_distribution}
            />
          </section>

          {/* Section 3: Triggered Business Rules */}
          <section aria-label="Business Rule Performance">
            <RuleAnalyticsTable
              rules={rules.rules}
              totalEvaluationsAnalyzed={rules.total_evaluations_analyzed}
            />
          </section>
        </div>
      ) : (
        <EmptyState
          title="No Analytics Data Available"
          description="No evaluation records found matching the selected time boundaries. Evaluate transactions to generate analytics."
          icon={<AlertTriangle size={32} />}
        />
      )}
    </div>
  );
};
