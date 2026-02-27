/**
 * Dashboard Screen – AI Signals Feed
 * Fetches batch signals, displays in FlatList with pull-to-refresh and skeleton.
 */
import React, { useState, useMemo } from 'react';
import {
    View, Text, FlatList, RefreshControl, StyleSheet,
    TextInput, TouchableOpacity, Platform,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useAuth } from '../../src/context/AuthContext';
import { useStockSignals } from '../../src/hooks/useStockData';
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

export default function DashboardScreen() {
    const { user, logout } = useAuth();
    const { signals, loading, refreshing, error, refresh } = useStockSignals(ALL_TICKERS);
    const [filter, setFilter] = useState('All');
    const [search, setSearch] = useState('');
    const insets = useSafeAreaInsets();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    const filtered = signals
        .filter((s) => filter === 'All' || s.signal === filter)
        .filter((s) => {
            const q = search.toLowerCase().trim();
            if (!q) return true;
            const ticker = s.symbol.toLowerCase();
            const name = (TICKER_TO_NAME[s.symbol] || '').toLowerCase();
            return ticker.includes(q) || name.includes(q);
        });

    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

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
                    <TouchableOpacity style={styles.logoutBtn} onPress={logout} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                        <Text style={styles.logoutIcon}>⏻</Text>
                    </TouchableOpacity>
                </View>
            </View>

            {/* Market pulse bar */}
            <View style={styles.pulseBar}>
                <PulseItem label="Signals" value={signals.length} styles={styles} />
                <View style={styles.pulseDivider} />
                <PulseItem label="BUY" value={signals.filter((s) => s.signal === 'BUY').length} valueColor={C.signal.buy} styles={styles} />
                <View style={styles.pulseDivider} />
                <PulseItem label="SELL" value={signals.filter((s) => s.signal === 'SELL').length} valueColor={C.signal.sell} styles={styles} />
                <View style={styles.pulseDivider} />
                <PulseItem label="HOLD" value={signals.filter((s) => s.signal === 'HOLD').length} valueColor={C.signal.hold} styles={styles} />
            </View>

            {/* Search */}
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
                    <TouchableOpacity onPress={() => setSearch('')}>
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
                        onPress={() => setFilter(f)}
                    >
                        <Text style={[styles.chipText, filter === f && styles.chipTextActive]}>{f}</Text>
                    </TouchableOpacity>
                ))}
            </View>

            {/* Loading */}
            {loading ? (
                <View style={styles.spinnerContainer}>
                    <LoadingSkeleton label="Fetching AI signals…" />
                </View>
            ) : error && signals.length === 0 ? (
                /* Full-screen error when nothing loaded at all */
                <View style={styles.errorFull}>
                    <Text style={styles.errorEmoji}>⚠️</Text>
                    <Text style={styles.errorTitle}>Could not load signals</Text>
                    <Text style={styles.errorMsg}>{error}</Text>
                    <TouchableOpacity style={styles.retryBtn} onPress={refresh}>
                        <Text style={styles.retryText}>Try Again</Text>
                    </TouchableOpacity>
                </View>
            ) : (
                <FlatList
                    data={filtered}
                    keyExtractor={(item) => item.symbol}
                    renderItem={({ item, index }) => <StockSignalCard item={item} index={index} />}
                    contentContainerStyle={styles.listPad}
                    ListEmptyComponent={
                        <EmptyState
                            emoji="🔍"
                            title={filter === 'All' ? 'No signals yet' : `No ${filter} signals`}
                            subtitle={filter === 'All' ? 'Pull down to refresh' : 'Try a different filter or pull down to refresh'}
                        />
                    }
                    ListHeaderComponent={
                        <>
                            {error && <ErrorBanner message={error} onRetry={refresh} />}
                            <SectionHeader title="AI Trade Signals" action="Refresh" onAction={refresh} />
                        </>
                    }
                    refreshControl={
                        <RefreshControl
                            refreshing={refreshing}
                            onRefresh={refresh}
                            tintColor={C.brand.purple}
                            colors={[C.brand.purple]}
                        />
                    }
                    showsVerticalScrollIndicator={false}
                />
            )}
        </View>
    );
}

function PulseItem({ label, value, valueColor, styles }) {
    return (
        <View style={styles.pulseItem}>
            <Text style={styles.pulseLabel}>{label}</Text>
            <Text style={[styles.pulseValue, valueColor && { color: valueColor }]}>{value}</Text>
        </View>
    );
}

function getGreeting() {
    const h = new Date().getHours();
    if (h < 12) return 'Morning';
    if (h < 17) return 'Afternoon';
    return 'Evening';
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
