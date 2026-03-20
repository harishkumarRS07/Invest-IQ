/**
 * useNewsData – custom hook for fetching and caching financial news.
 *
 * Features:
 *  - Module-level TTL cache keyed by ticker symbol (survives tab switches)
 *  - Cache-first rendering: shows cached data instantly, refetches in background
 *    when stale (> TTL)
 *  - Pull-to-refresh bypasses cache
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { newsApi } from '../services/api';

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

// Module-level cache – persists across unmount/remount of the component
const newsCache = new Map(); // ticker → { articles, timestamp }

export function useNewsData(ticker) {
    // Seed state from cache immediately (avoids loading spinner if data is fresh)
    const getCachedArticles = () => {
        const entry = newsCache.get(ticker);
        if (entry && Date.now() - entry.timestamp < CACHE_TTL_MS) {
            return entry.articles;
        }
        return [];
    };

    const [articles, setArticles] = useState(getCachedArticles);
    const [loading, setLoading] = useState(() => {
        // Skip loading state if we already have fresh cache
        const entry = newsCache.get(ticker);
        return !(entry && Date.now() - entry.timestamp < CACHE_TTL_MS);
    });
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const activeTickerRef = useRef(ticker);

    const loadNews = useCallback(async (isRefresh = false) => {
        const currentTicker = ticker;
        activeTickerRef.current = currentTicker;

        // Check cache on non-refresh fetch
        if (!isRefresh) {
            const cached = newsCache.get(currentTicker);
            if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
                setArticles(cached.articles);
                setLoading(false);
                return;
            }
        }

        isRefresh ? setRefreshing(true) : setLoading(true);
        setError(null);

        try {
            const result = await newsApi.getNews(currentTicker);
            const fetched = result.articles || [];

            // Only update state if ticker hasn't changed while awaiting
            if (activeTickerRef.current === currentTicker) {
                newsCache.set(currentTicker, { articles: fetched, timestamp: Date.now() });
                setArticles(fetched);
            }
        } catch (err) {
            if (activeTickerRef.current === currentTicker) {
                setError(err.message);
            }
        } finally {
            if (activeTickerRef.current === currentTicker) {
                setLoading(false);
                setRefreshing(false);
            }
        }
    }, [ticker]);

    // Reload when ticker changes
    useEffect(() => {
        // Reset to cached data (or empty) for the new ticker immediately
        setArticles(getCachedArticles());
        loadNews();
    }, [loadNews]);

    const refresh = useCallback(() => loadNews(true), [loadNews]);

    return { articles, loading, refreshing, error, refresh };
}
