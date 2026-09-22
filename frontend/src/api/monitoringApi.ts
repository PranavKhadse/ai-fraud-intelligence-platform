/**
 * Monitoring API Client for Phase 13.6 ML & Model Monitoring.
 * Communicates with backend endpoints under /api/v1/monitoring.
 */

import { apiClient } from './client.ts';
import type {
  FeatureDriftDetailResponse,
  FeatureDriftListResponse,
  FeatureDriftQueryParams,
  ModelPerformanceResponse,
  MonitoringHealthResponse,
  MonitoringQueryParams,
  MonitoringSnapshotListResponse,
  PerformanceQueryParams,
  PredictionDriftResponse,
  SnapshotQueryParams,
} from '../types/monitoring.ts';

/**
 * Fetch high-level monitoring health status, component statuses, and derived active alerts.
 */
export async function fetchMonitoringHealth(
  params: MonitoringQueryParams = {}
): Promise<MonitoringHealthResponse> {
  const query = new URLSearchParams();
  if (params.window) query.set('window', params.window);
  if (params.start_time) query.set('start_time', params.start_time);
  if (params.end_time) query.set('end_time', params.end_time);
  if (params.model_version) query.set('model_version', params.model_version);

  const qs = query.toString();
  const endpoint = qs ? `/api/v1/monitoring/health?${qs}` : '/api/v1/monitoring/health';
  return apiClient<MonitoringHealthResponse>(endpoint);
}

/**
 * Fetch 55-feature drift report with optional category, status, search, and sorting filters.
 */
export async function fetchFeatureDriftList(
  params: FeatureDriftQueryParams = {}
): Promise<FeatureDriftListResponse> {
  const query = new URLSearchParams();
  if (params.window) query.set('window', params.window);
  if (params.start_time) query.set('start_time', params.start_time);
  if (params.end_time) query.set('end_time', params.end_time);
  if (params.category) query.set('category', params.category);
  if (params.status) query.set('status', params.status);
  if (params.search_term && params.search_term.trim()) {
    query.set('search_term', params.search_term.trim());
  }
  if (params.sort_by) query.set('sort_by', params.sort_by);
  if (params.limit !== undefined) query.set('limit', String(params.limit));
  if (params.offset !== undefined) query.set('offset', String(params.offset));
  if (params.model_version) query.set('model_version', params.model_version);

  const qs = query.toString();
  const endpoint = qs ? `/api/v1/monitoring/drift/features?${qs}` : '/api/v1/monitoring/drift/features';
  return apiClient<FeatureDriftListResponse>(endpoint);
}

/**
 * Fetch granular single-feature drift report and baseline vs current distribution comparisons.
 */
export async function fetchFeatureDriftDetail(
  featureName: string,
  params: MonitoringQueryParams = {}
): Promise<FeatureDriftDetailResponse> {
  const query = new URLSearchParams();
  if (params.window) query.set('window', params.window);
  if (params.start_time) query.set('start_time', params.start_time);
  if (params.end_time) query.set('end_time', params.end_time);
  if (params.model_version) query.set('model_version', params.model_version);

  const qs = query.toString();
  const encodedName = encodeURIComponent(featureName);
  const endpoint = qs
    ? `/api/v1/monitoring/drift/features/${encodedName}?${qs}`
    : `/api/v1/monitoring/drift/features/${encodedName}`;
  return apiClient<FeatureDriftDetailResponse>(endpoint);
}

/**
 * Fetch prediction score drift, 10-bucket risk score drift, tier/action JSD, and override rate shifts.
 */
export async function fetchPredictionDrift(
  params: MonitoringQueryParams = {}
): Promise<PredictionDriftResponse> {
  const query = new URLSearchParams();
  if (params.window) query.set('window', params.window);
  if (params.start_time) query.set('start_time', params.start_time);
  if (params.end_time) query.set('end_time', params.end_time);
  if (params.model_version) query.set('model_version', params.model_version);

  const qs = query.toString();
  const endpoint = qs ? `/api/v1/monitoring/drift/predictions?${qs}` : '/api/v1/monitoring/drift/predictions';
  return apiClient<PredictionDriftResponse>(endpoint);
}

/**
 * Fetch ground-truth performance metrics at operating threshold (default 0.78), operational metrics,
 * review queue purity, and mathematical degradation alerts.
 */
export async function fetchModelPerformance(
  params: PerformanceQueryParams = {}
): Promise<ModelPerformanceResponse> {
  const query = new URLSearchParams();
  if (params.window) query.set('window', params.window);
  if (params.start_time) query.set('start_time', params.start_time);
  if (params.end_time) query.set('end_time', params.end_time);
  if (params.operating_threshold !== undefined) {
    query.set('operating_threshold', String(params.operating_threshold));
  }
  if (params.model_version) query.set('model_version', params.model_version);

  const qs = query.toString();
  const endpoint = qs ? `/api/v1/monitoring/performance?${qs}` : '/api/v1/monitoring/performance';
  return apiClient<ModelPerformanceResponse>(endpoint);
}

/**
 * Fetch paginated list of persisted historical monitoring snapshots.
 */
export async function fetchMonitoringSnapshots(
  params: SnapshotQueryParams = {}
): Promise<MonitoringSnapshotListResponse> {
  const query = new URLSearchParams();
  if (params.model_version) query.set('model_version', params.model_version);
  if (params.window_type) query.set('window_type', params.window_type);
  if (params.status) query.set('status', params.status);
  if (params.start_time) query.set('start_time', params.start_time);
  if (params.end_time) query.set('end_time', params.end_time);
  if (params.limit !== undefined) query.set('limit', String(params.limit));
  if (params.offset !== undefined) query.set('offset', String(params.offset));

  const qs = query.toString();
  const endpoint = qs ? `/api/v1/monitoring/snapshots?${qs}` : '/api/v1/monitoring/snapshots';
  return apiClient<MonitoringSnapshotListResponse>(endpoint);
}
