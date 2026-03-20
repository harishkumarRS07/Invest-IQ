/**
 * Dashboard Screen – AI Signals Feed (Optimized)
 *
 * Performance improvements:
 *  - Debounced ticker search (300ms) via useDebounce → avoids filtering on
 *    every keystroke
 *  - useMemo for filtered list and pulse bar counts → computed only when
 *    inputs change, not on every render
 *  - useCallback for renderItem, keyExtractor, filter chip handlers, logout
 *  - FlatList virtualization: initialNumToRender, maxToRenderPerBatch,
 *    windowSize, updateCellsBatchingPeriod, removeClippedSubviews
 *  - PulseItem extracted and memoized with React.memo
 */
import React, { useState, useMemo, useCallback } from 'react';
import {
    View, Text, FlatList, RefreshControl, StyleSheet,
    TextInput, TouchableOpacity, Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useAuth } from '../../src/context/AuthContext';
import { useStockSignals } from '../../src/hooks/useStockData';
import { useDebounce } from '../../src/hooks/useDebounce';
import StockSignalCard from '../../src/components/StockSignalCard';
import {
    LoadingSkeleton, ErrorBanner, EmptyState, SectionHeader,
} from '../../src/components/ui';
import { Spacing, Radius, Typography } from '../../src/constants/theme';
import { useColors } from '../../src/context/ThemeContext';

const ALL_TICKERS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'];
const FILTERS = ['All', 'BUY', 'SELL', 'HOLD'];

// Map ticker → full company name so users can search by name OR ticker
const TICKER_TO_NAME = {
    RELIANCE: 'Reliance Industries',
    TCS: 'Tata Consultancy Services',
    INFY: 'Infosys',
    HDFCBANK: 'HDFC Bank',
    ICICIBANK: 'ICICI Bank',
};

function getGreeting() {
    const h = new Date().getHours();
    if (h < 12) return 'Morning';
    if (h < 17) return 'Afternoon';
    return 'Evening';
}

// ── Memoized sub-component ────────────────────────────────────────────────────
const PulseItem = React.memo(({ label, value, valueColor, styles }) => (
    <View style={styles.pulseItem}>
        <Text style={styles.pulseLabel}>{label}</Text>
        <Text style={[styles.pulseValue, valueColor && { color: valueColor }]}>{value}</Text>
    </View>
));

// ── Main screen ───────────────────────────────────────────────────────────────
export default function DashboardScreen() {
    const { user, logout } = useAuth();
    const { signals, loading, refreshing, error, refresh } = useStockSignals(ALL_TICKERS);
    const [filter, setFilter] = useState('All');
    const [search, setSearch] = useState('');
    const insets = useSafeAreaInsets();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    // Debounce search input: filtering only recalculates 300ms after user stops typing
    const debouncedSearch = useDebounce(search, 300);

    // Memoized pulse counts – avoids re-computing on every render tick
    const pulseCounts = useMemo(() => ({
        total: signals.length,
        buy: signals.filter((s) => s.signal === 'BUY').length,
        sell: signals.filter((s) => s.signal === 'SELL').length,
        hold: signals.filter((s) => s.signal === 'HOLD').length,
    }), [signals]);

    // Memoized filtered list – only recomputed when signals/filter/debouncedSearch change
    const filtered = useMemo(() => {
        const q = debouncedSearch.toLowerCase().trim();
        return signals.filter((s) => {
            // Signal filter
            if (filter !== 'All' && s.signal !== filter) return false;
            // Search filter
            if (q) {
                const ticker = s.symbol.toLowerCase();
                const name = (TICKER_TO_NAME[s.symbol] || '').toLowerCase();
                if (!ticker.includes(q) && !name.includes(q)) return false;
            }
            return true;
        });
    }, [signals, filter, debouncedSearch]);

    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

    // Stable callbacks
    const renderItem = useCallback(
        ({ item, index }) => <StockSignalCard item={item} index={index} />,
        []
    );
    const keyExtractor = useCallback((item) => item.symbol, []);
    const handleClearSearch = useCallback(() => setSearch(''), []);
    const handleLogout = useCallback(() => logout(), [logout]);
    const handleRefresh = useCallback(() => refresh(), [refresh]);

    // Memoized filter chip handlers (one per chip, stable across renders)
    const filterHandlers = useMemo(
        () => Object.fromEntries(FILTERS.map((f) => [f, () => setFilter(f)])),
        []
    );

    const listHeader = useMemo(() => (
        <>
            {error && <ErrorBanner message={error} onRetry={refresh} />}
            <SectionHeader title="AI Trade Signals" action="Refresh" onAction={refresh} />
        </>
    ), [error, refresh]);

    const listEmpty = useMemo(() => (
        <EmptyState
            emoji="🔍"
            title={filter === 'All' ? 'No signals yet' : `No ${filter} signals`}
            subtitle={filter === 'All' ? 'Pull down to refresh' : 'Try a different filter or pull down to refresh'}
        />
    ), [filter]);

    return (
        <View style={[styles.root, { paddingTop: insets.top }]}>
            {/* Header */}
            <View style={styles.header}>
                <View>
                    <Text style={styles.greeting}>Good {getGreeting()} 👋</Text>
                    <Text style={styles.userName}>{user?.name || 'Trader'}</Text>
                </View>
                <View style={styles.headerRight}>
                    <Text style={styles.time}>{timeStr}</Text>
                    <TouchableOpacity
                        style={styles.logoutBtn}
                        onPress={handleLogout}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Text style={styles.logoutIcon}>⏻</Text>
                    </TouchableOpacity>
                </View>
            </View>

            {/* Market pulse bar – uses memoized counts */}
            <View style={styles.pulseBar}>
                <PulseItem label="Signals" value={pulseCounts.total} styles={styles} />
                <View style={styles.pulseDivider} />
                <PulseItem label="BUY" value={pulseCounts.buy} valueColor={C.signal.buy} styles={styles} />
                <View style={styles.pulseDivider} />
                <PulseItem label="SELL" value={pulseCounts.sell} valueColor={C.signal.sell} styles={styles} />
                <View style={styles.pulseDivider} />
                <PulseItem label="HOLD" value={pulseCounts.hold} valueColor={C.signal.hold} styles={styles} />
            </View>

            {/* Search – debounced */}
            <View style={styles.searchRow}>
                <Text style={styles.searchIcon}>🔍</Text>
                <TextInput
                    style={styles.searchInput}
                    placeholder="Search by name or ticker…"
                    placeholderTextColor={C.text.muted}
                    value={search}
                    onChangeText={setSearch}
                    autoCapitalize="none"
                    autoCorrect={false}
                />
                {search.length > 0 && (
                    <TouchableOpacity onPress={handleClearSearch}>
                        <Text style={styles.clearIcon}>✕</Text>
                    </TouchableOpacity>
                )}
            </View>

            {/* Filter chips */}
            <View style={styles.filterRow}>
                {FILTERS.map((f) => (
                    <TouchableOpacity
                        key={f}
                        style={[styles.chip, filter === f && styles.chipActive]}
                        onPress={filterHandlers[f]}
                    >
                        <Text style={[styles.chipText, filter === f && styles.chipTextActive]}>{f}</Text>
                    </TouchableOpacity>
                ))}
            </View>

            {/* Content */}
            {loading ? (
                <View style={styles.spinnerContainer}>
                    <LoadingSkeleton label="Fetching AI signals…" />
                </View>
            ) : error && signals.length === 0 ? (
                <View style={styles.errorFull}>
                    <Text style={styles.errorEmoji}>⚠️</Text>
                    <Text style={styles.errorTitle}>Could not load signals</Text>
                    <Text style={styles.errorMsg}>{error}</Text>
                    <TouchableOpacity style={styles.retryBtn} onPress={handleRefresh}>
                        <Text style={styles.retryText}>Try Again</Text>
                    </TouchableOpacity>
                </View>
            ) : (
                <FlatList
                    data={filtered}
                    keyExtractor={keyExtractor}
                    renderItem={renderItem}
                    contentContainerStyle={[styles.listPad, { paddingBottom: insets.bottom + Spacing.xxl }]}
                    ListEmptyComponent={listEmpty}
                    ListHeaderComponent={listHeader}
                    refreshControl={
                        <RefreshControl
                            refreshing={refreshing}
                            onRefresh={handleRefresh}
                            tintColor={C.brand.purple}
                            colors={[C.brand.purple]}
                        />
                    }
                    showsVerticalScrollIndicator={false}
                    // ── Virtualization tuning ──────────────────────────────
                    initialNumToRender={5}
                    maxToRenderPerBatch={8}
                    updateCellsBatchingPeriod={50}
                    windowSize={10}
                    removeClippedSubviews={Platform.OS === 'android'}
                // ─────────────────────────────────────────────────────
                />
            )}
        </View>
    );
}

const makeStyles = (C) => StyleSheet.create({
    root: { flex: 1, backgroundColor: C.bg.primary },
    spinnerContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: Spacing.lg,
        paddingVertical: Spacing.md,
    },
    greeting: { color: C.text.secondary, fontSize: Typography.sizes.sm },
    userName: {
        color: C.text.primary, fontSize: Typography.sizes.xl,
        fontWeight: Typography.weights.bold, marginTop: 1,
    },
    headerRight: { alignItems: 'flex-end', gap: 6 },
    time: { color: C.text.secondary, fontSize: Typography.sizes.sm },
    logoutBtn: {
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.full,
        width: 34, height: 34,
        alignItems: 'center', justifyContent: 'center',
        borderWidth: 1, borderColor: C.border.default,
    },
    logoutIcon: { fontSize: 14 },
    pulseBar: {
        flexDirection: 'row',
        backgroundColor: C.bg.card,
        marginHorizontal: Spacing.lg,
        borderRadius: Radius.lg,
        paddingVertical: Spacing.md,
        paddingHorizontal: Spacing.sm,
        marginBottom: Spacing.md,
        borderWidth: 1, borderColor: C.border.default,
        justifyContent: 'space-around', alignItems: 'center',
    },
    pulseItem: { alignItems: 'center', flex: 1 },
    pulseLabel: { color: C.text.muted, fontSize: Typography.sizes.xs },
    pulseValue: {
        color: C.text.primary,
        fontSize: Typography.sizes.xl,
        fontWeight: Typography.weights.bold,
        marginTop: 2,
    },
    pulseDivider: { width: 1, height: 28, backgroundColor: C.border.default },
    searchRow: {
        flexDirection: 'row', alignItems: 'center',
        backgroundColor: C.bg.input,
        marginHorizontal: Spacing.lg,
        borderRadius: Radius.md,
        paddingHorizontal: Spacing.md,
        height: 44, marginBottom: Spacing.sm,
        borderWidth: 1, borderColor: C.border.default,
    },
    searchIcon: { fontSize: 15, marginRight: 8 },
    searchInput: { flex: 1, color: C.text.primary, fontSize: Typography.sizes.md },
    clearIcon: { color: C.text.muted, fontSize: 14, paddingLeft: 8 },
    filterRow: {
        flexDirection: 'row', paddingHorizontal: Spacing.lg,
        marginBottom: Spacing.sm, gap: Spacing.sm,
    },
    chip: {
        paddingHorizontal: Spacing.md, paddingVertical: 6,
        borderRadius: Radius.full,
        backgroundColor: C.bg.elevated,
        borderWidth: 1, borderColor: C.border.default,
    },
    chipActive: { backgroundColor: C.brand.glow, borderColor: C.brand.purple },
    chipText: { color: C.text.secondary, fontSize: Typography.sizes.sm, fontWeight: Typography.weights.medium },
    chipTextActive: { color: C.brand.purple, fontWeight: Typography.weights.bold },
    listPad: {
        paddingHorizontal: Spacing.lg,
        paddingBottom: Spacing.xxl,
    },
    errorFull: {
        flex: 1, alignItems: 'center', justifyContent: 'center',
        paddingHorizontal: Spacing.xl, paddingTop: Spacing.xxl,
    },
    errorEmoji: { fontSize: 48, marginBottom: Spacing.md },
    errorTitle: {
        color: C.text.primary,
        fontSize: Typography.sizes.xl,
        fontWeight: Typography.weights.bold,
        textAlign: 'center', marginBottom: Spacing.sm,
    },
    errorMsg: {
        color: C.text.secondary,
        fontSize: Typography.sizes.sm,
        textAlign: 'center', lineHeight: 20,
        marginBottom: Spacing.xl,
    },
    retryBtn: {
        backgroundColor: C.brand.purple,
        borderRadius: Radius.lg,
        paddingHorizontal: Spacing.xl,
        paddingVertical: 14,
    },
    retryText: {
        color: '#fff',
        fontWeight: Typography.weights.bold,
        fontSize: Typography.sizes.md,
    },
});
