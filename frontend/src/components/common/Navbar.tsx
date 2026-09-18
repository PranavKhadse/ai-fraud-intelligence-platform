import React from 'react';
import { ShieldCheck, RefreshCw, Activity, TrendingUp, Sliders, Inbox } from 'lucide-react';
import { SystemStatusBadge } from './SystemStatusBadge.tsx';
import { ActorContextSwitcher } from '../cases/ActorContextSwitcher.tsx';
import type { HealthResponse } from '../../types/api.ts';

export type DashboardTab = 'operations' | 'analytics' | 'simulator' | 'cases';

interface NavbarProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  activeTab?: DashboardTab;
  unassignedCaseCount?: number;
  onTabChange?: (tab: DashboardTab) => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  health,
  healthLoading,
  healthError,
  activeTab = 'operations',
  unassignedCaseCount,
  onTabChange,
  onRefresh,
  isRefreshing,
}) => {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <div className="navbar-logo">
          <ShieldCheck size={22} />
        </div>
        <div>
          <div className="navbar-title">Fraud Intelligence</div>
          <div className="navbar-subtitle">Risk Intelligence &amp; Decision Platform</div>
        </div>
      </div>

      {onTabChange && (
        <nav className="navbar-nav-tabs" aria-label="Main Navigation">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'operations' ? 'active' : ''}`}
            onClick={() => onTabChange('operations')}
            aria-selected={activeTab === 'operations'}
            role="tab"
          >
            <Activity size={15} />
            <span>Live Operations</span>
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'cases' ? 'active' : ''}`}
            onClick={() => onTabChange('cases')}
            aria-selected={activeTab === 'cases'}
            role="tab"
          >
            <Inbox size={15} />
            <span>Case Review</span>
            {unassignedCaseCount !== undefined && unassignedCaseCount > 0 && (
              <span className="nav-tab-counter-badge" title={`${unassignedCaseCount} unassigned cases`}>
                {unassignedCaseCount}
              </span>
            )}
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => onTabChange('analytics')}
            aria-selected={activeTab === 'analytics'}
            role="tab"
          >
            <TrendingUp size={15} />
            <span>Analytics &amp; Trends</span>
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
            onClick={() => onTabChange('simulator')}
            aria-selected={activeTab === 'simulator'}
            role="tab"
          >
            <Sliders size={15} />
            <span>What-If Simulator</span>
          </button>
        </nav>
      )}

      <div className="navbar-actions">
        <ActorContextSwitcher />
        <SystemStatusBadge
          health={health}
          loading={healthLoading}
          error={healthError}
        />
        {onRefresh && (
          <button
            type="button"
            onClick={onRefresh}
            disabled={isRefreshing}
            className="navbar-refresh-btn"
            title="Refresh overview metrics"
            style={{
              background: 'transparent',
              border: '1px solid var(--border-card)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.375rem 0.625rem',
              color: 'var(--text-secondary)',
              cursor: isRefreshing ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.375rem',
              fontSize: '0.75rem',
              fontWeight: 500,
            }}
          >
            <RefreshCw
              size={14}
              style={{
                animation: isRefreshing ? 'spin 1s linear infinite' : 'none',
              }}
            />
            <span>Refresh</span>
          </button>
        )}
      </div>
    </header>
  );
};

