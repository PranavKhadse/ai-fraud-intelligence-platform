/**
 * ML & Model Monitoring Type Definitions for Phase 13.6.
 * Matches backend Pydantic v2 schemas from Phase 13.5 (/api/v1/monitoring).
 */

export type DriftSeverity = 'NORMAL' | 'WARNING' | 'CRITICAL' | 'INSUFFICIENT_DATA';
export type MetricConfidence = 'NORMAL_CONFIDENCE' | 'LOW_SAMPLE' | 'INSUFFICIENT_DATA';
export type MonitoringTimeWindow = '1h' | '24h' | '7d' | '30d' | 'custom';
export type FeatureCategory = 'amount' | 'velocity' | 'geographic' | 'demographic' | 'behavioral';
export type FeatureSortKey = 'psi' | 'ks_stat' | 'missing_delta';

export interface MonitoringAlertItem {
  component: 'data_drift' | 'prediction_drift' | 'performance' | string;
  feature_name: string | null;
  severity: 'WARNING' | 'CRITICAL' | string;
  metric_name: string;
  observed_value: number;
  threshold_value: number | null;
  message: string;
}

export interface MonitoringHealthResponse {
  model_version: string;
  window_type: string;
  window_start: string;
  window_end: string;
  sample_count: number;
  labeled_count: number;
  overall_status: DriftSeverity;
  data_drift_status: DriftSeverity;
  prediction_drift_status: DriftSeverity;
  performance_status: DriftSeverity;
  active_alert_count: number;
  active_alerts: MonitoringAlertItem[];
  created_at: string;
}

export interface FeatureDriftItemResponse {
  feature_name: string;
  feature_type: 'numerical' | 'categorical' | string;
  category: FeatureCategory | string;
  status: DriftSeverity;
  psi: number;
  ks_statistic: number | null;
  ks_p_value: number | null;
  js_divergence: number | null;
  missing_rate_current: number;
  missing_rate_delta: number;
  unseen_category_rate: number | null;
}

export interface FeatureDriftListResponse {
  model_version: string;
  window_type: string;
  window_start: string;
  window_end: string;
  sample_count: number;
  overall_status: DriftSeverity;
  total_features_count: number;
  drifted_features_count: number;
  critical_features_count: number;
  warning_features_count: number;
  items: FeatureDriftItemResponse[];
}

export interface FeatureDriftDetailResponse {
  feature_name: string;
  feature_type: 'numerical' | 'categorical' | string;
  category: FeatureCategory | string;
  status: DriftSeverity;
  psi: number;
  ks_statistic: number | null;
  ks_p_value: number | null;
  js_divergence: number | null;
  missing_rate_baseline: number;
  missing_rate_current: number;
  missing_rate_delta: number;
  unseen_category_rate: number | null;
  unseen_categories: string[];
  baseline_distribution: Record<string, number>;
  current_distribution: Record<string, number>;
  bin_edges: number[] | null;
}

export interface PredictionDriftResponse {
  model_version: string;
  window_type: string;
  window_start: string;
  window_end: string;
  sample_count: number;
  overall_status: DriftSeverity;
  model_score_psi: number;
  model_score_status: DriftSeverity;
  model_score_mean: number;
  model_score_std: number;
  risk_score_psi: number;
  risk_score_status: DriftSeverity;
  risk_tier_jsd: number;
  risk_tier_status: DriftSeverity;
  action_jsd: number;
  action_status: DriftSeverity;
  override_rate_current: number;
  override_rate_baseline: number;
  override_rate_delta: number;
  model_score_distribution: Record<string, number>;
  risk_score_buckets_distribution: Record<string, number>;
  risk_tier_distribution: Record<string, number>;
  action_distribution: Record<string, number>;
  active_alerts: MonitoringAlertItem[];
}

export interface ConfusionMatrixResponse {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  total: number;
}

export interface ThresholdMetricsResponse {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  fpr: number;
  tpr: number;
  confusion_matrix: ConfusionMatrixResponse;
}

export interface OperationalMetricsResponse {
  decision_precision_block: number;
  decision_recall_intervention: number;
  review_queue_purity: number;
  total_reviews_count: number;
  fraud_in_review_count: number;
  total_blocks_count: number;
  fraud_in_block_count: number;
}

export interface MetricDegradationItemResponse {
  metric_name: string;
  observed_value: number;
  baseline_value: number;
  relative_delta: number;
  absolute_delta: number;
  severity: DriftSeverity;
  confidence: MetricConfidence;
  alert_message: string | null;
}

export interface ModelPerformanceResponse {
  model_version: string;
  window_type: string;
  window_start: string;
  window_end: string;
  dataset_row_count: number;
  labeled_sample_count: number;
  fraud_cases_count: number;
  legitimate_cases_count: number;
  suspicious_resolved_count: number;
  overall_performance_status: DriftSeverity;
  confidence: MetricConfidence;
  operating_threshold: number;
  primary_metrics: ThresholdMetricsResponse;
  comparison_metrics: ThresholdMetricsResponse;
  pr_auc: number | null;
  roc_auc: number | null;
  operational_metrics: OperationalMetricsResponse;
  degradation_results: Record<string, MetricDegradationItemResponse>;
  active_degradation_alerts: MonitoringAlertItem[];
}

export interface MonitoringSnapshotItemResponse {
  id: string;
  model_version: string;
  window_type: 'HOURLY' | 'DAILY' | string;
  window_start: string;
  window_end: string;
  sample_count: number;
  labeled_count: number;
  overall_status: DriftSeverity;
  data_drift_status: DriftSeverity;
  prediction_drift_status: DriftSeverity;
  performance_status: DriftSeverity;
  created_at: string;
}

export interface MonitoringSnapshotListResponse {
  total_count: number;
  limit: number;
  offset: number;
  items: MonitoringSnapshotItemResponse[];
}

export interface MonitoringQueryParams {
  window?: MonitoringTimeWindow;
  start_time?: string;
  end_time?: string;
  model_version?: string;
}

export interface FeatureDriftQueryParams extends MonitoringQueryParams {
  category?: string;
  status?: string;
  search_term?: string;
  sort_by?: FeatureSortKey;
  limit?: number;
  offset?: number;
}

export interface PerformanceQueryParams extends MonitoringQueryParams {
  operating_threshold?: number;
}

export interface SnapshotQueryParams {
  model_version?: string;
  window_type?: string;
  status?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
  offset?: number;
}
