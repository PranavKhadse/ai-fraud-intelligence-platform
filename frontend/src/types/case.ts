/**
 * Case Management & Human Review Lifecycle Type Definitions.
 * Matches backend Pydantic v2 schemas from Phase 12.3.
 */

import type {
  DecisionAction,
  FeatureAttributionDetail,
  PolicyMode,
  ReasonCodeDetail,
  RiskTier,
  RuleMatchDetail,
  RuleOutcome,
} from './api.ts';

export type AuditActorType = 'SYSTEM' | 'ANALYST' | 'ADMIN' | 'API_CLIENT';
export type CaseStatus = 'OPEN' | 'IN_REVIEW' | 'ESCALATED' | 'RESOLVED' | 'CLOSED';
export type CasePriority = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
export type CaseDisposition =
  | 'CONFIRMED_FRAUD'
  | 'FALSE_POSITIVE'
  | 'LEGITIMATE'
  | 'SUSPICIOUS_RESOLVED';
export type CaseTriggerSource =
  | 'AUTOMATED_REVIEW_POLICY'
  | 'AUTOMATED_RULE_OVERRIDE'
  | 'MANUAL_ANALYST_ESCALATION';
export type CaseNoteType = 'INVESTIGATION' | 'ESCALATION' | 'DISPOSITION' | 'SYSTEM_AUDIT';

export interface CaseResponse {
  id: string;
  case_number: string;
  transaction_id: string;
  evaluation_id: string;
  status: CaseStatus;
  priority: CasePriority;
  trigger_source: CaseTriggerSource;
  assigned_to: string | null;
  assigned_at: string | null;
  opened_at: string;
  resolved_at: string | null;
  closed_at: string | null;
  disposition: CaseDisposition | null;
  disposition_reason: string | null;
  dispositioned_by: string | null;
  dispositioned_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface CaseQueueItem {
  id: string;
  case_number: string;
  transaction_id: string;
  evaluation_id: string;
  status: CaseStatus;
  priority: CasePriority;
  trigger_source: CaseTriggerSource;
  assigned_to: string | null;
  assigned_at: string | null;
  opened_at: string;
  resolved_at: string | null;
  disposition: CaseDisposition | null;
  account_id: string;
  amount: number;
  currency: string;
  merchant_category: string;
  risk_score: number;
  risk_tier: RiskTier;
  decision_action: DecisionAction;
  model_score: number;
}

export interface CaseListResponse {
  items: CaseQueueItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface CaseSummaryResponse {
  total_open: number;
  unassigned_count: number;
  in_review_count: number;
  escalated_count: number;
  resolved_today: number;
  resolved_last_24h: number;
  critical_priority_count: number;
}

export interface CaseNoteItem {
  id: string;
  case_id: string;
  author_id: string;
  author_role: AuditActorType;
  note_type: CaseNoteType;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface CaseTransactionContext {
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
  city_pop?: number | null;
  transaction_timestamp: string;
  features_snapshot: Record<string, unknown> | null;
}

export interface CaseEvaluationContext {
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
  evaluated_at: string;
  rule_matches: RuleMatchDetail[];
  reason_codes: ReasonCodeDetail[];
  feature_attributions: FeatureAttributionDetail[];
}

export interface CaseDetailResponse {
  case: CaseResponse;
  transaction: CaseTransactionContext;
  evaluation: CaseEvaluationContext;
  notes: CaseNoteItem[];
  notes_total: number;
}

export interface TimelineEventItem {
  id: string;
  event_type: string;
  action: string;
  actor_type: AuditActorType;
  actor_id: string | null;
  correlation_id: string | null;
  payload: Record<string, unknown> | null;
  event_timestamp: string;
}

export interface CaseTimelineResponse {
  case_id: string;
  case_number: string;
  events: TimelineEventItem[];
  total_events: number;
}

export interface CreateCaseRequest {
  transaction_id: string;
  initial_note: string;
  priority?: CasePriority | null;
  evaluation_id?: string | null;
}

export interface CaseAssignmentRequest {
  action: 'CLAIM' | 'ASSIGN' | 'UNASSIGN';
  assignee_id?: string | null;
  reason?: string | null;
}

export interface CaseStatusUpdateRequest {
  target_status: CaseStatus;
  reason: string;
}

export interface CreateCaseNoteRequest {
  content: string;
  note_type?: CaseNoteType;
}

export interface CaseDispositionRequest {
  disposition: CaseDisposition;
  reason: string;
}

export interface CaseQueryParams {
  limit?: number;
  offset?: number;
  status?: CaseStatus;
  priority?: CasePriority;
  assigned_to?: string;
  risk_tier?: RiskTier;
  min_score?: number;
  max_score?: number;
  search_term?: string;
  start_date?: string;
  end_date?: string;
  sort_by?: 'opened_at' | 'priority' | 'risk_score';
  sort_order?: 'asc' | 'desc';
}
