/**
 * Base HTTP API Client using native fetch with structured error handling.
 */

export class ApiClientError extends Error {
  statusCode: number;
  data?: unknown;

  constructor(message: string, statusCode: number, data?: unknown) {
    super(message);
    this.name = 'ApiClientError';
    this.statusCode = statusCode;
    this.data = data;
  }
}

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const config: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(endpoint, config);

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      let errorData: unknown;
      try {
        errorData = await response.json();
        if (errorData && typeof errorData === 'object' && 'detail' in errorData) {
          errorMessage = String((errorData as { detail: unknown }).detail);
        }
      } catch {
        // Response wasn't JSON, retain statusText
      }
      throw new ApiClientError(errorMessage, response.status, errorData);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiClientError) {
      throw error;
    }
    const message = error instanceof Error ? error.message : 'Unknown network error';
    throw new ApiClientError(`Network connection error: ${message}`, 0);
  }
}
