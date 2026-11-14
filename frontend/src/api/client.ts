// Types mirror api/schemas.py. Phase 5 replaces this hand-written copy with
// types generated from FastAPI's OpenAPI schema, so the two cannot drift.

export interface SourceStatus {
  role: 'primary' | 'backup' | 'crypto';
  name: string;
  configured: boolean;
}

export interface StoredSeries {
  symbol: string;
  interval: string;
  n_bars: number;
  first_timestamp: string;
  last_timestamp: string;
}

export interface AccountStatus {
  configured: boolean;
  paper: boolean | null;
  ok: boolean | null;
  error: string | null;
  balances: Record<string, number> | null;
}

export interface StatusResponse {
  sources: SourceStatus[];
  store: StoredSeries[];
  account: AccountStatus;
  settings: Record<string, Record<string, unknown> | string>;
}

export interface ParameterInfo {
  name: string;
  type: 'integer' | 'number' | 'string' | 'object';
  required: boolean;
  default: unknown;
  description: string;
  label: string;
  help: string;
  advanced: boolean;
  minimum: number | null;
  maximum: number | null;
  step: number | null;
}

export interface StrategyInfo {
  id: string;
  label: string;
  summary: string;
  kind: 'pair' | 'cross_sectional';
  title: string;
  explanation: string;
  parameters: ParameterInfo[];
}

export interface Universe {
  id: string;
  label: string;
  description: string;
  symbols: string[];
}

export interface UniversesResponse {
  universes: Universe[];
  pairs: { a: string; b: string; label: string }[];
}

export interface ValidationResult {
  valid: boolean;
  errors: Record<string, string>;
  error: string | null;
  params: Record<string, unknown> | null;
  symbols: string[] | null;
}

export interface SetupIn {
  name: string;
  strategy_id: string;
  params: Record<string, unknown>;
  symbols: string[];
}

export interface Setup extends SetupIn {
  id: string;
  created_at: string;
  updated_at: string;
}

/** An HTTP error that keeps FastAPI's `detail` so forms can show field errors. */
export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === 'string' ? detail : `Request failed (${status})`);
  }
}

/** Call an endpoint under /api, returning parsed JSON (or undefined for 204). */
async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 204) return undefined as T;
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, payload?.detail ?? response.statusText);
  return payload as T;
}

export const getJson = <T>(path: string) => request<T>('GET', path);
export const postJson = <T>(path: string, body: unknown) => request<T>('POST', path, body);
export const putJson = <T>(path: string, body: unknown) => request<T>('PUT', path, body);
export const deleteJson = (path: string) => request<void>('DELETE', path);
