/**
 * News Screen – Optimized
 *
 * Performance improvements:
 *  - useNewsData hook: module-level TTL cache per ticker (instant render on tab revisit)
 *  - useCallback for loadNews, renderItem, keyExtractor, onSelectTicker
 *  - FlatList virtualization props: maxToRenderPerBatch, windowSize,
 *    initialNumToRender, removeClippedSubviews, updateCellsBatchingPeriod
 *  - Stable getItemLayout for fixed-height cards (eliminates layout scanning)
 */
import React, { useState, useCallback, useMemo } from 'react';
import { View, Text, FlatList, StyleSheet, RefreshControl, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LoadingSkeleton, ErrorBanner } from '../../src/components/ui';
import { Spacing, Typography } from '../../src/constants/theme';
import { useColors } from '../../src/context/ThemeContext';
import { useNewsData } from '../../src/hooks/useNewsData';

import NewsCard from '../../src/components/news/NewsCard';
import TickerFilter from '../../src/components/news/TickerFilter';

// Active tickers – defined outside component so it's a stable reference
const TICKERS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'TATAMOTORS'];

// Approximate card height for getItemLayout (eliminates measure-on-scroll)
const CARD_HEIGHT = 130; // title(~66) + header(~28) + footer(~20) + padding

export default function NewsScreen() {
    const insets = useSafeAreaInsets();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    const [selectedTicker, setSelectedTicker] = useState(TICKERS[0]);

    // Cache-backed hook – no re-fetch if data is fresh on ticker change
    const { articles, loading, refreshing, error, refresh } = useNewsData(selectedTicker);

    // Stable handler passed to memoized TickerFilter
    const handleSelectTicker = useCallback((ticker) => {
        setSelectedTicker(ticker);
    }, []);

    // Stable renderItem – prevents FlatList from re-rendering all items
    const renderItem = useCallback(({ item }) => <NewsCard article={item} />, []);

    // Stable keyExtractor
    const keyExtractor = useCallback(
        (item, index) => item.id || item.title || String(index),
        []
    );

    // getItemLayout: tells FlatList exact item dimensions → enables
    // fast scroll-to-index and skips layout measurement per cell
    const getItemLayout = useCallback(
        (_data, index) => ({
            length: CARD_HEIGHT,
            offset: CARD_HEIGHT * index,
            index,
        }),
        []
    );

    const handleRefresh = useCallback(() => refresh(), [refresh]);

    const renderHeader = useMemo(() => (
        <View style={styles.header}>
            <Text style={styles.pageTitle}>News</Text>
            <Text style={styles.pageSubtitle}>📰 Real-time Market Intelligence</Text>
        </View>
    ), [styles]);

    const renderEmpty = useMemo(() => {
        if (loading) return <LoadingSkeleton rows={6} />;
        return (
            <Text style={styles.emptyText}>
                No news found for {selectedTicker} recently.
            </Text>
        );
    }, [loading, selectedTicker, styles]);

    return (
        <View style={[styles.root, { paddingTop: insets.top }]}>
            {renderHeader}

            <View style={styles.filterWrapper}>
                <TickerFilter
                    tickers={TICKERS}
                    selectedTicker={selectedTicker}
                    onSelectTicker={handleSelectTicker}
                />
            </View>

            {error && <ErrorBanner message={error} onRetry={refresh} />}

            <FlatList
                data={articles}
                keyExtractor={keyExtractor}
                renderItem={renderItem}
                getItemLayout={getItemLayout}
                contentContainerStyle={[styles.listContent, { paddingBottom: insets.bottom + Spacing.xxl }]}
                showsVerticalScrollIndicator={false}
                ListEmptyComponent={renderEmpty}
                // ── Virtualization tuning ──────────────────────────────────
                initialNumToRender={5}
                maxToRenderPerBatch={8}
                updateCellsBatchingPeriod={50}
                windowSize={10}
                removeClippedSubviews={Platform.OS === 'android'}
                // ─────────────────────────────────────────────────────────
                refreshControl={
                    <RefreshControl
                        refreshing={refreshing}
                        onRefresh={handleRefresh}
                        tintColor={C.brand.purple}
                    />
                }
            />
        </View>
    );
}

const makeStyles = (C) => StyleSheet.create({
    root: { flex: 1, backgroundColor: C.bg.primary },
    header: {
        paddingHorizontal: Spacing.lg,
        paddingTop: Spacing.lg,
        paddingBottom: Spacing.sm,
    },
    pageTitle: {
        color: C.text.primary,
        fontSize: Typography.sizes.xxl,
        fontWeight: Typography.weights.black,
    },
    pageSubtitle: {
        color: C.text.secondary,
        fontSize: Typography.sizes.md,
        marginTop: 4,
    },
    filterWrapper: {
        marginBottom: Spacing.md,
    },
    listContent: {
        paddingHorizontal: Spacing.lg,
    },
    emptyText: {
        color: C.text.muted,
        fontSize: Typography.sizes.md,
        textAlign: 'center',
        marginTop: Spacing.xl,
    },
});
