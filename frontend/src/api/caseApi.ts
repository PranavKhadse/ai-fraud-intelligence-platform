import { apiClient } from './client.ts';
import type {
  CaseAssignmentRequest,
  CaseDetailResponse,
  CaseDispositionRequest,
  CaseListResponse,
  CaseNoteItem,
  CaseQueryParams,
  CaseResponse,
  CaseStatusUpdateRequest,
  CaseSummaryResponse,
  CaseTimelineResponse,
  CreateCaseNoteRequest,
  CreateCaseRequest,
  AuditActorType,
} from '../types/case.ts';

export interface DevActorContext {
  actorId: string;
  actorRole: AuditActorType;
}

const DEV_ACTOR_STORAGE_KEY_ID = 'fraud_dev_actor_id';
const DEV_ACTOR_STORAGE_KEY_ROLE = 'fraud_dev_actor_role';

/**
 * Retrieve active Development/Test Actor Context from browser storage.
 * Defaults to 'analyst_01' (ANALYST).
 */
export function getDevActorContext(): DevActorContext {
  try {
    const actorId = localStorage.getItem(DEV_ACTOR_STORAGE_KEY_ID) || 'analyst_01';
    const rawRole = localStorage.getItem(DEV_ACTOR_STORAGE_KEY_ROLE) || 'ANALYST';
    const actorRole: AuditActorType = ['ANALYST', 'ADMIN', 'SYSTEM', 'API_CLIENT'].includes(rawRole)
      ? (rawRole as AuditActorType)
      : 'ANALYST';
    return { actorId, actorRole };
  } catch {
    return { actorId: 'analyst_01', actorRole: 'ANALYST' };
  }
}

/**
 * Persist active Development/Test Actor Context to browser storage.
 */
export function setDevActorContext(actor: DevActorContext): void {
  try {
    localStorage.setItem(DEV_ACTOR_STORAGE_KEY_ID, actor.actorId);
    localStorage.setItem(DEV_ACTOR_STORAGE_KEY_ROLE, actor.actorRole);
  } catch {
    // Storage access unavailable
  }
}

/**
 * Determine if development actor header injection is permitted by the environment.
 * Disabled if VITE_ALLOW_DEV_ACTOR_HEADERS is explicitly 'false' or in production mode without explicit override.
 */
export function isDevActorHeadersAllowed(): boolean {
  const envVal = import.meta.env?.VITE_ALLOW_DEV_ACTOR_HEADERS;
  if (envVal === 'false') {
    return false;
  }
  if (import.meta.env?.PROD && envVal !== 'true') {
    return false;
  }
  return true;
}

/**
 * Generate request headers containing the caller's development/test actor identity.
 * Returns empty object {} if dev actor injection is disabled (fail-closed in production).
 */
export function getDevActorHeaders(): Record<string, string> {
  if (!isDevActorHeadersAllowed()) {
    return {};
  }
  const { actorId, actorRole } = getDevActorContext();
  return {
    'X-Actor-ID': actorId,
    'X-Actor-Role': actorRole,
  };
}

/**
 * Fetch paginated review queue cases with multi-dimensional filtering.
 */

export async function fetchCases(
  params: CaseQueryParams = {}
): Promise<CaseListResponse> {
  const query = new URLSearchParams();

  if (params.limit !== undefined) query.set('limit', String(params.limit));
  if (params.offset !== undefined) query.set('offset', String(params.offset));
  if (params.status) query.set('status', params.status);
  if (params.priority) query.set('priority', params.priority);
  if (params.assigned_to) query.set('assigned_to', params.assigned_to);
  if (params.risk_tier) query.set('risk_tier', params.risk_tier);
  if (params.min_score !== undefined) query.set('min_score', String(params.min_score));
  if (params.max_score !== undefined) query.set('max_score', String(params.max_score));
  if (params.search_term && params.search_term.trim()) query.set('search_term', params.search_term.trim());
  if (params.start_date) query.set('start_date', params.start_date);
  if (params.end_date) query.set('end_date', params.end_date);
  if (params.sort_by) query.set('sort_by', params.sort_by);
  if (params.sort_order) query.set('sort_order', params.sort_order);

  const queryString = query.toString();
  const endpoint = queryString ? `/api/v1/cases?${queryString}` : '/api/v1/cases';

  return apiClient<CaseListResponse>(endpoint, {
    headers: getDevActorHeaders(),
  });
}

/**
 * Fetch queue summary KPI counts (open, unassigned, in review, escalated, resolved, critical).
 */
export async function fetchCaseSummary(): Promise<CaseSummaryResponse> {
  return apiClient<CaseSummaryResponse>('/api/v1/cases/summary', {
    headers: getDevActorHeaders(),
  });
}

/**
 * Fetch complete investigation detail for a single case.
 */
export async function fetchCaseDetail(
  caseId: string,
  notesLimit: number = 100
): Promise<CaseDetailResponse> {
  const cleanId = encodeURIComponent(caseId.trim());
  return apiClient<CaseDetailResponse>(`/api/v1/cases/${cleanId}?notes_limit=${notesLimit}`, {
    headers: getDevActorHeaders(),
  });
}

/**
 * Manually escalate a transaction into a human review case.
 */
export async function createManualCase(
  payload: CreateCaseRequest
): Promise<CaseResponse> {
  return apiClient<CaseResponse>('/api/v1/cases', {
    method: 'POST',
    headers: getDevActorHeaders(),
    body: JSON.stringify(payload),
  });
}

/**
 * Mutate case assignment (CLAIM, ASSIGN, UNASSIGN).
 */
export async function updateCaseAssignment(
  caseId: string,
  payload: CaseAssignmentRequest
): Promise<CaseResponse> {
  const cleanId = encodeURIComponent(caseId.trim());
  return apiClient<CaseResponse>(`/api/v1/cases/${cleanId}/assignment`, {
    method: 'PATCH',
    headers: getDevActorHeaders(),
    body: JSON.stringify(payload),
  });
}

/**
 * Mutate case lifecycle status (ESCALATE, CLOSE, REOPEN).
 */
export async function updateCaseStatus(
  caseId: string,
  payload: CaseStatusUpdateRequest
): Promise<CaseResponse> {
  const cleanId = encodeURIComponent(caseId.trim());
  return apiClient<CaseResponse>(`/api/v1/cases/${cleanId}/status`, {
    method: 'PATCH',
    headers: getDevActorHeaders(),
    body: JSON.stringify(payload),
  });
}

/**
 * Fetch chronological notes list for a case.
 */
export async function fetchCaseNotes(
  caseId: string,
  limit: number = 100,
  offset: number = 0
): Promise<CaseNoteItem[]> {
  const cleanId = encodeURIComponent(caseId.trim());
  return apiClient<CaseNoteItem[]>(`/api/v1/cases/${cleanId}/notes?limit=${limit}&offset=${offset}`, {
    headers: getDevActorHeaders(),
  });
}

/**
 * Append an investigation note to a case.
 */
export async function createCaseNote(
  caseId: string,
  payload: CreateCaseNoteRequest
): Promise<CaseNoteItem> {
  const cleanId = encodeURIComponent(caseId.trim());
  return apiClient<CaseNoteItem>(`/api/v1/cases/${cleanId}/notes`, {
    method: 'POST',
    headers: getDevActorHeaders(),
    body: JSON.stringify(payload),
  });
}

/**
 * Record human review outcome and resolve a case.
 */
export async function submitCaseDisposition(
  caseId: string,
  payload: CaseDispositionRequest
): Promise<CaseResponse> {
  const cleanId = encodeURIComponent(caseId.trim());
  return apiClient<CaseResponse>(`/api/v1/cases/${cleanId}/disposition`, {
    method: 'POST',
    headers: getDevActorHeaders(),
    body: JSON.stringify(payload),
  });
}

/**
 * Fetch chronological audit timeline events for a case.
 */
export async function fetchCaseTimeline(
  caseId: string,
  limit: number = 100
): Promise<CaseTimelineResponse> {
  const cleanId = encodeURIComponent(caseId.trim());
  return apiClient<CaseTimelineResponse>(`/api/v1/cases/${cleanId}/timeline?limit=${limit}`, {
    headers: getDevActorHeaders(),
  });
}
