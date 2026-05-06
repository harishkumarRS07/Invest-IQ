/**
 * useStockData - Custom hook for fetching and caching stock signal data.
 * Wraps stockApi.batchSignals with loading, error, and refresh state.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { stockApi } from '../services/api';

const DEFAULT_TICKERS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'];
const CACHE_DURATION_MS = 5 * 60 * 1000; // 5 minutes

export function useStockSignals(tickers = DEFAULT_TICKERS) {
    const [signals, setSignals] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const lastFetchRef = useRef(null);

    const fetch = useCallback(async (isRefresh = false) => {
        // Stale check
        const now = Date.now();
        if (!isRefresh && lastFetchRef.current && now - lastFetchRef.current < CACHE_DURATION_MS) {
            return;
        }
        isRefresh ? setRefreshing(true) : setLoading(true);
        setError(null);
        try {
            const data = await stockApi.batchSignals(tickers);
            console.log('[useStockSignals] Fetched data:', data);
            // Ensure data is an array
            const signalsArray = Array.isArray(data) ? data : (data?.signals && Array.isArray(data.signals) ? data.signals : []);
            console.log('[useStockSignals] Final signals array:', signalsArray, 'count:', signalsArray.length);
            setSignals(signalsArray);
            lastFetchRef.current = Date.now();
        } catch (err) {
            console.error('[useStockSignals] Error:', err.message);
            setError(err.message);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, [tickers.join(',')]);

    useEffect(() => { fetch(); }, [fetch]);

    const refresh = useCallback(() => fetch(true), [fetch]);

    return { signals, loading, refreshing, error, refresh };
}

/**
 * useStockDetail - Fetch a full prediction for a single stock.
 */
export function useStockDetail(symbol) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetch = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await stockApi.predict(symbol);
            setData(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [symbol]);

    useEffect(() => { fetch(); }, [fetch]);

    return { data, loading, error, refetch: fetch };
}
