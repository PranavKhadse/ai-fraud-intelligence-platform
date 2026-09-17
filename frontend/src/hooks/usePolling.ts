import { useEffect, useRef, useState, useCallback } from 'react';

export interface UsePollingOptions {
  interval?: number; // in milliseconds (default 5000)
  enabled?: boolean; // default true
  immediate?: boolean; // trigger immediately on mount (default true)
}

export interface UsePollingReturn<T> {
  data: T | null;
  loading: boolean;
  isPolling: boolean;
  error: string | null;
  lastUpdated: Date | null;
  isAutoPolling: boolean;
  interval: number;
  setInterval: (interval: number) => void;
  toggleAutoPolling: () => void;
  setAutoPolling: (enabled: boolean) => void;
  refresh: () => Promise<void>;
}

export function usePolling<T>(
  fetchFn: () => Promise<T>,
  options: UsePollingOptions = {}
): UsePollingReturn<T> {
  const {
    interval: defaultInterval = 5000,
    enabled: defaultEnabled = true,
    immediate = true,
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(immediate);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isAutoPolling, setIsAutoPolling] = useState<boolean>(defaultEnabled);
  const [intervalTime, setIntervalTime] = useState<number>(defaultInterval);

  // In-flight request lock & mounted check
  const inFlightRef = useRef<boolean>(false);
  const isMountedRef = useRef<boolean>(true);
  const fetchFnRef = useRef(fetchFn);

  useEffect(() => {
    fetchFnRef.current = fetchFn;
  }, [fetchFn]);

  const executeFetch = useCallback(async (isInitial = false) => {
    // Prevent overlapping in-flight requests
    if (inFlightRef.current) {
      return;
    }

    inFlightRef.current = true;
    if (isInitial) {
      setLoading(true);
    } else {
      setIsPolling(true);
    }

    try {
      const result = await fetchFnRef.current();
      if (isMountedRef.current) {
        setData(result);
        setError(null);
        setLastUpdated(new Date());
      }
    } catch (err) {
      if (isMountedRef.current) {
        const msg = err instanceof Error ? err.message : 'Failed to fetch data';
        setError(msg);
      }
    } finally {
      inFlightRef.current = false;
      if (isMountedRef.current) {
        setLoading(false);
        setIsPolling(false);
      }
    }
  }, []);

  // Initial trigger
  useEffect(() => {
    isMountedRef.current = true;
    if (immediate) {
      executeFetch(true);
    }
    return () => {
      isMountedRef.current = false;
    };
  }, [immediate, executeFetch]);

  // Polling interval timer
  useEffect(() => {
    if (!isAutoPolling) {
      return;
    }

    const timerId = window.setInterval(() => {
      executeFetch(false);
    }, intervalTime);

    return () => {
      window.clearInterval(timerId);
    };
  }, [isAutoPolling, intervalTime, executeFetch]);

  const toggleAutoPolling = useCallback(() => {
    setIsAutoPolling((prev) => !prev);
  }, []);

  const refresh = useCallback(async () => {
    await executeFetch(false);
  }, [executeFetch]);

  return {
    data,
    loading,
    isPolling,
    error,
    lastUpdated,
    isAutoPolling,
    interval: intervalTime,
    setInterval: setIntervalTime,
    toggleAutoPolling,
    setAutoPolling: setIsAutoPolling,
    refresh,
  };
}
