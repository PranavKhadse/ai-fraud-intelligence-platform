/**
 * Core API Schema Type Definitions for Fraud Intelligence Platform.
 * Matches backend Pydantic models.
 */

export type DecisionAction = 'APPROVE' | 'REVIEW' | 'BLOCK';
export type RiskTier = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type PolicyMode = 'TRI_TIER' | 'BINARY_AUTO';
export type RuleOutcome = 'APPROVE' | 'REVIEW' | 'BLOCK' | 'MONITOR';
export type RuleType = 'VELOCITY' | 'AMOUNT' | 'GEOGRAPHY' | 'COMPLIANCE' | 'BEHAVIORAL' | 'CUSTOM';
export type AttributionDirection = 'RISK_INCREASING' | 'MITIGATING';
export type ReasonSource = 'MODEL' | 'RULE';
export type ReasonSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';

export interface DashboardOverviewResponse {
  total_transactions: number;
  total_amount: number;
  approval_count: number;
  approval_rate: number;
  review_count: number;
  review_rate: number;
  block_count: number;
  block_rate: number;
  average_risk_score: number;
  average_latency_ms: number;
}

export interface TransactionListItem {
  id: string;
  external_transaction_id: string | null;
  transaction_timestamp: string;
  amount: number;
  currency: string;
  merchant_category: string;
  risk_score: number;
  risk_tier: RiskTier;
  decision_action: DecisionAction;
  is_overridden: boolean;
  evaluation_latency_ms: number | null;
}

export interface TransactionListResponse {
  items: TransactionListItem[];
  total_count: number;
  limit: number;
  offset: number;
}

export interface TransactionQueryParams {
  limit?: number;
  offset?: number;
  decision_action?: DecisionAction;
  risk_tier?: RiskTier;
  search_term?: string;
  min_score?: number;
  max_score?: number;
  start_date?: string;
  end_date?: string;
}

export interface TransactionMetadataDetail {
  id: string;
  external_transaction_id: string | null;
  account_id: string;
  merchant_id: string | null;
  merchant_category: string;
  job_category: string;
  amount: number;
  currency: string;
  cardholder_lat: number;
  cardholder_long: number;
  merchant_lat: number;
  merchant_long: number;
  city_pop: number;
  transaction_timestamp: string;
  created_at: string;
}

export interface EvaluationDetail {
  id: string;
  model_version: string;
  policy_mode: PolicyMode;
  model_score: number;
  risk_score: number;
  risk_tier: RiskTier;
  decision_action: DecisionAction;
  baseline_action: DecisionAction;
  is_overridden: boolean;
  rule_action: RuleOutcome | null;
  decision_reason: string;
  output_margin: number | null;
  base_value: number | null;
  evaluation_latency_ms: number | null;
  correlation_id: string | null;
  evaluated_at: string;
}

export interface FeatureAttributionDetail {
  id: string;
  feature_name: string;
  display_name: string;
  raw_value: unknown;
  shap_value: number;
  direction: AttributionDirection;
  relative_contribution_pct: number;
  rank: number;
}

export interface ReasonCodeDetail {
  id: string;
  code: string;
  headline: string;
  description: string;
  category: string;
  source: ReasonSource;
  severity: ReasonSeverity;
  rank: number;
}

export interface RuleMatchDetail {
  id: string;
  rule_id: string;
  description: string;
  feature_name: string;
  operator: string;
  comparison_value: string;
  outcome: RuleOutcome;
  rule_type: RuleType;
  priority: number;
}

export interface AuditLogDetail {
  id: string;
  event_type: string;
  action: string;
  actor_type: string;
  actor_id: string | null;
  correlation_id: string | null;
  client_ip: string | null;
  event_timestamp: string;
}

export interface TransactionDetailResponse {
  transaction: TransactionMetadataDetail;
  evaluation: EvaluationDetail | null;
  features: Record<string, unknown>;
  feature_attributions: FeatureAttributionDetail[];
  reason_codes: ReasonCodeDetail[];
  rule_matches: RuleMatchDetail[];
  audit_trail: AuditLogDetail[];
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  model_loaded: boolean;
  model_version: string;
  rules_loaded_count: number;
  timestamp: string;
}

export interface ApiError {
  message: string;
  statusCode?: number;
}

export interface DistributionBucket {
  bucket_label: string;
  lower_bound: number;
  upper_bound: number;
  count: number;
  percentage: number;
}

export interface CategoryCount {
  category: string;
  count: number;
  percentage: number;
}

export interface AnalyticsDistributionsResponse {
  total_evaluated: number;
  risk_score_distribution: DistributionBucket[];
  model_score_distribution: DistributionBucket[];
  risk_tier_distribution: CategoryCount[];
  decision_distribution: CategoryCount[];
}

export type TrendInterval = 'hourly' | 'daily';

export interface TrendDataPoint {
  timestamp: string;
  total_count: number;
  total_amount: number;
  average_risk_score: number;
  approval_count: number;
  review_count: number;
  block_count: number;
  high_critical_count: number;
}

export interface AnalyticsTrendsResponse {
  interval: TrendInterval;
  start_date: string;
  end_date: string;
  data_points: TrendDataPoint[];
}

export interface RuleOutcomeBreakdown {
  outcome: RuleOutcome;
  count: number;
  percentage: number;
}

export interface RuleAnalyticsItem {
  rule_id: string;
  description: string;
  rule_type: RuleType;
  priority: number;
  trigger_count: number;
  affected_transactions: number;
  trigger_rate: number;
  override_count: number;
  outcomes: RuleOutcomeBreakdown[];
}

export interface AnalyticsRulesResponse {
  total_rules_active: number;
  total_evaluations_analyzed: number;
  rules: RuleAnalyticsItem[];
}

export interface AnalyticsQueryParams {
  start_date?: string;
  end_date?: string;
  decision_action?: DecisionAction;
  risk_tier?: RiskTier;
  interval?: TrendInterval;
  limit?: number;
}

export interface SimulationRequest {
  baseline_transaction_id?: string;
  simulated_features: Record<string, any>;
  top_k?: number;
  top_mitigating?: number;
  max_reasons?: number;
}

export interface FeatureDiffItem {
  feature_name: string;
  display_name: string;
  category: string;
  baseline_value: any;
  simulated_value: any;
  is_modified: boolean;
  delta?: number | null;
}

export type RuleDiffStatus = 'NEWLY_TRIGGERED' | 'RESOLVED' | 'PERSISTENT' | 'NEITHER';

export interface RuleDiffItem {
  rule_id: string;
  description: string;
  rule_type: string;
  priority: number;
  outcome: string;
  baseline_triggered: boolean;
  simulated_triggered: boolean;
  diff_status: RuleDiffStatus;
}

export interface BaselineEvaluationSummary {
  transaction_id: string;
  external_transaction_id: string;
  risk_score: number;
  risk_tier: RiskTier;
  decision_action: DecisionAction;
  model_score: number;
  is_overridden: boolean;
  rules_triggered_count: number;
}

export interface SimulatedEvaluationSummary {
  risk_score: number;
  risk_tier: RiskTier;
  decision_action: DecisionAction;
  model_score: number;
  base_value?: number | null;
  output_margin?: number | null;
  is_overridden: boolean;
  decision_reason: string;
  rule_matches: RuleMatchDetail[];
  reason_codes: ReasonCodeDetail[];
  feature_attributions: FeatureAttributionDetail[];
}

export interface SimulationComparisonSummary {
  risk_score_delta: number;
  model_score_delta: number;
  tier_changed: boolean;
  action_changed: boolean;
  modified_features_count: number;
  feature_diffs: FeatureDiffItem[];
  rule_diffs: RuleDiffItem[];
}

export interface SimulationResponse {
  is_simulation: boolean;
  simulated_at: string;
  evaluation_latency_ms: number;
  baseline_transaction_id?: string | null;
  baseline?: BaselineEvaluationSummary | null;
  simulated: SimulatedEvaluationSummary;
  comparison?: SimulationComparisonSummary | null;
}



