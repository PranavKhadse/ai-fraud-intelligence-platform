import { apiClient } from './client.ts';
import type {
  AnalyticsDistributionsResponse,
  AnalyticsQueryParams,
  AnalyticsRulesResponse,
  AnalyticsTrendsResponse,
  DashboardOverviewResponse,
  HealthResponse,
  SimulationRequest,
  SimulationResponse,
  TransactionDetailResponse,
  TransactionListResponse,
  TransactionQueryParams,
} from '../types/api.ts';

/**
 * Fetch high-level aggregated operational KPI metrics from the backend.
 */
export async function fetchDashboardOverview(): Promise<DashboardOverviewResponse> {
  return apiClient<DashboardOverviewResponse>('/api/v1/dashboard/overview');
}

/**
 * Fetch active service operational readiness and model telemetry.
 */
export async function fetchHealthStatus(): Promise<HealthResponse> {
  return apiClient<HealthResponse>('/api/v1/health');
}

/**
 * Fetch paginated transaction records with optional filters.
 */
export async function fetchTransactions(
  params: TransactionQueryParams = {}
): Promise<TransactionListResponse> {
  const query = new URLSearchParams();

  if (params.limit !== undefined) query.set('limit', String(params.limit));
  if (params.offset !== undefined) query.set('offset', String(params.offset));
  if (params.decision_action) query.set('decision_action', params.decision_action);
  if (params.risk_tier) query.set('risk_tier', params.risk_tier);
  if (params.search_term && params.search_term.trim()) query.set('search_term', params.search_term.trim());
  if (params.min_score !== undefined) query.set('min_score', String(params.min_score));
  if (params.max_score !== undefined) query.set('max_score', String(params.max_score));
  if (params.start_date) query.set('start_date', params.start_date);
  if (params.end_date) query.set('end_date', params.end_date);

  const queryString = query.toString();
  const endpoint = queryString
    ? `/api/v1/dashboard/transactions?${queryString}`
    : '/api/v1/dashboard/transactions';

  return apiClient<TransactionListResponse>(endpoint);
}

/**
 * Fetch complete transaction deep investigation context by internal UUID or external client ID.
 */
export async function fetchTransactionDetail(
  transactionId: string
): Promise<TransactionDetailResponse> {
  const cleanId = encodeURIComponent(transactionId.trim());
  return apiClient<TransactionDetailResponse>(`/api/v1/dashboard/transactions/${cleanId}`);
}

/**
 * Fetch fraud risk score histograms, probability distributions, and tier/decision breakdowns.
 */
export async function fetchAnalyticsDistributions(
  params: AnalyticsQueryParams = {}
): Promise<AnalyticsDistributionsResponse> {
  const query = new URLSearchParams();
  if (params.start_date) query.set('start_date', params.start_date);
  if (params.end_date) query.set('end_date', params.end_date);
  if (params.decision_action) query.set('decision_action', params.decision_action);
  if (params.risk_tier) query.set('risk_tier', params.risk_tier);

  const queryString = query.toString();
  const endpoint = queryString
    ? `/api/v1/dashboard/analytics/distributions?${queryString}`
    : '/api/v1/dashboard/analytics/distributions';

  return apiClient<AnalyticsDistributionsResponse>(endpoint);
}

/**
 * Fetch time-series fraud risk trends for volume, monetary totals, scores, and decisions.
 */
export async function fetchAnalyticsTrends(
  params: AnalyticsQueryParams = {}
): Promise<AnalyticsTrendsResponse> {
  const query = new URLSearchParams();
  if (params.start_date) query.set('start_date', params.start_date);
  if (params.end_date) query.set('end_date', params.end_date);
  if (params.interval) query.set('interval', params.interval);

  const queryString = query.toString();
  const endpoint = queryString
    ? `/api/v1/dashboard/analytics/trends?${queryString}`
    : '/api/v1/dashboard/analytics/trends';

  return apiClient<AnalyticsTrendsResponse>(endpoint);
}

/**
 * Fetch ranked business rule triggers, affected transactions, override counts, and outcome breakdown.
 */
export async function fetchAnalyticsRules(
  params: AnalyticsQueryParams = {}
): Promise<AnalyticsRulesResponse> {
  const query = new URLSearchParams();
  if (params.start_date) query.set('start_date', params.start_date);
  if (params.end_date) query.set('end_date', params.end_date);
  if (params.limit !== undefined) query.set('limit', String(params.limit));

  const queryString = query.toString();
  const endpoint = queryString
    ? `/api/v1/dashboard/analytics/rules?${queryString}`
    : '/api/v1/dashboard/analytics/rules';

  return apiClient<AnalyticsRulesResponse>(endpoint);
}

/**
 * Execute in-memory what-if fraud risk simulation without database persistence.
 */
export async function simulateTransactionScenario(
  payload: SimulationRequest
): Promise<SimulationResponse> {
  return apiClient<SimulationResponse>(
    '/api/v1/dashboard/simulate',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    }
  );
}




