/**
 * Portfolio Screen
 * Displays optimized portfolio allocation and key metrics from the backend.
 */
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    View, Text, ScrollView, StyleSheet, RefreshControl,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { PieChart } from 'react-native-chart-kit';
import { Dimensions } from 'react-native';
import { portfolioApi } from '../../src/services/api';
import {
    Card, LoadingSkeleton, ErrorBanner, SectionHeader, IndicatorRow,
} from '../../src/components/ui';
import { Spacing, Radius, Typography } from '../../src/constants/theme';
import { useColors } from '../../src/context/ThemeContext';

const W = Dimensions.get('window').width;

// Symbols sent to backend — backend appends .NS internally when fetching
const ALL_SYMBOLS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'];

// Fixed chart colors for consistent, vibrant allocation display
const CHART_COLORS = ['#7B61FF', '#00D07C', '#FF5353', '#F5A623', '#4D9DE0'];

// ─── Metric label mapping (matches backend portfolio.py keys exactly) ─────────
const METRIC_LABELS = {
    expected_annual_return: 'Expected Return (Annual)',
    annual_volatility: 'Volatility (Annual)',
    sharpe_ratio: 'Sharpe Ratio',
    max_drawdown: 'Max Drawdown',
};

function formatMetricLabel(key) {
    return (
        METRIC_LABELS[key] ||
        key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
    );
}

/**
 * Format a metric value for display.
 * Sharpe ratio is dimensionless (show 2dp). Return/volatility are fractions → show as %.
 */
function formatMetricValue(key, val) {
    if (typeof val !== 'number') return String(val);
    if (key === 'sharpe_ratio') return val.toFixed(2);
    // Return & volatility come as fractions (e.g. 0.15 = 15%)
    return (val * 100).toFixed(2) + '%';
}

/** Strip the .NS suffix that yfinance appends to Indian tickers */
function stripNS(symbol) {
    return symbol.replace(/\.NS$/i, '');
}

export default function PortfolioScreen() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const insets = useSafeAreaInsets();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    const load = useCallback(async (isRefresh = false) => {
        isRefresh ? setRefreshing(true) : setLoading(true);
        setError(null);
        try {
            const result = await portfolioApi.optimize(ALL_SYMBOLS);
            setData(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const pieData = useMemo(() => {
        if (!data?.allocation) return [];
        return Object.entries(data.allocation)
            .sort(([, a], [, b]) => b - a)
            .map(([name, pct], i) => {
                const population = Math.round(pct * 10000) / 100; // 0-100%
                return {
                    name: stripNS(name),
                    population: population > 0 ? population : 0.01, // never exactly 0
                    color: CHART_COLORS[i % CHART_COLORS.length],
                    legendFontColor: C.text.secondary,
                    legendFontSize: 12,
                };
            })
            .filter((item) => item.population >= 0.01); // hide truly zero-weight items
    }, [data, C.text.secondary]);

    // Sorted allocation entries for the bar section — same order as pie, exclude zeros
    const sortedAllocation = useMemo(() => {
        if (!data?.allocation) return [];
        return Object.entries(data.allocation)
            .sort(([, a], [, b]) => b - a)
            .filter(([, w]) => w > 0.0001); // exclude effectively-zero weights
    }, [data]);

    const chartWidth = W - Spacing.lg * 2 - Spacing.md * 2;

    return (
        <View style={[styles.root, { paddingTop: insets.top }]}>
            <ScrollView
                contentContainerStyle={[styles.scroll, { paddingBottom: insets.bottom + Spacing.xxl }]}
                showsVerticalScrollIndicator={false}
                refreshControl={
                    <RefreshControl
                        refreshing={refreshing}
                        onRefresh={() => load(true)}
                        tintColor={C.brand.purple}
                        colors={[C.brand.purple]}
                    />
                }
            >
                {/* ── Header ─────────────────────────────────────────────── */}
                <View style={styles.pageHeader}>
                    <Text style={styles.pageTitle}>Portfolio</Text>
                    <Text style={styles.pageSubtitle}>💼 AI-Optimised Allocation</Text>
                </View>

                {error && <ErrorBanner message={error} onRetry={() => load()} />}

                {loading ? (
                    <LoadingSkeleton rows={8} />
                ) : data ? (
                    <>
                        {/* ── Pie Chart ──────────────────────────────────── */}
                        {pieData.length > 0 && (
                            <Card style={styles.chartCardWrapper}>
                                <SectionHeader title="Optimal Allocation" />
                                <View style={styles.chartRow}>
                                    <View style={styles.pieWrapper}>
                                        <PieChart
                                            data={pieData}
                                            width={160} // Just large enough for the pie
                                            height={160}
                                            chartConfig={{
                                                color: (opacity = 1) => `rgba(123, 97, 255, ${opacity})`,
                                            }}
                                            accessor="population"
                                            backgroundColor="transparent"
                                            paddingLeft={35} // center to fit bounding box
                                            center={[0, 0]}
                                            absolute={false}
                                            hasLegend={false}
                                        />
                                    </View>

                                    <View style={styles.legendContainer}>
                                        {pieData.map((item, idx) => (
                                            <View key={item.name} style={styles.legendItem}>
                                                <View style={[styles.legendColor, { backgroundColor: item.color }]} />
                                                <Text style={styles.legendText} numberOfLines={1}>
                                                    {Math.round(item.population)}% {item.name}
                                                </Text>
                                            </View>
                                        ))}
                                    </View>
                                </View>
                            </Card>
                        )}

                        {/* ── Allocation Bars ─────────────────────────────── */}
                        <Card>
                            <SectionHeader title="Stock Weights" />
                            {sortedAllocation.map(([symbol, weight], i) => {
                                const label = stripNS(symbol);
                                const pctNum = Math.round(weight * 10000) / 100; // e.g. 32.14
                                const barPct = Math.min(Math.max(pctNum, 0), 100);
                                return (
                                    <View key={symbol} style={styles.allocRow}>
                                        <View style={styles.allocLeft}>
                                            <View
                                                style={[
                                                    styles.allocDot,
                                                    { backgroundColor: CHART_COLORS[i % CHART_COLORS.length] },
                                                ]}
                                            />
                                            <Text style={styles.allocName} numberOfLines={1}>
                                                {label}
                                            </Text>
                                        </View>
                                        <View style={styles.allocBarWrap}>
                                            <View
                                                style={[
                                                    styles.allocBar,
                                                    {
                                                        width: `${barPct}%`,
                                                        backgroundColor: CHART_COLORS[i % CHART_COLORS.length],
                                                    },
                                                ]}
                                            />
                                        </View>
                                        <Text style={styles.allocPct}>{pctNum.toFixed(1)}%</Text>
                                    </View>
                                );
                            })}
                        </Card>

                        {/* ── Portfolio Metrics ───────────────────────────── */}
                        {data.metrics && Object.keys(data.metrics).length > 0 && (
                            <Card>
                                <SectionHeader title="Performance Metrics" />
                                {Object.entries(data.metrics).map(([key, val]) => (
                                    <IndicatorRow
                                        key={key}
                                        label={formatMetricLabel(key)}
                                        value={formatMetricValue(key, val)}
                                        unit=""
                                    />
                                ))}
                            </Card>
                        )}

                        {/* ── Disclaimer ─────────────────────────────────── */}
                        <View style={styles.disclaimer}>
                            <Text style={styles.disclaimerText}>
                                ⚠️ AI-generated allocation for financial advice.
                            </Text>
                        </View>
                    </>
                ) : null}
            </ScrollView>
        </View>
    );
}

const makeStyles = (C) => StyleSheet.create({
    root: { flex: 1, backgroundColor: C.bg.primary },
    scroll: { padding: Spacing.lg, paddingBottom: Spacing.xxl },
    pageHeader: { marginBottom: Spacing.lg },
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

    chartCardWrapper: {
        overflow: 'hidden',
    },
    chartRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginTop: Spacing.sm,
    },
    pieWrapper: {
        width: 160,
        height: 160,
        alignItems: 'center',
        justifyContent: 'center',
        marginLeft: -10, // Slight shift to make layout breathe better
    },
    legendContainer: {
        flex: 1,
        justifyContent: 'center',
        marginLeft: Spacing.lg,
    },
    legendItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 10,
    },
    legendColor: {
        width: 14,
        height: 14,
        borderRadius: Radius.full,
        marginRight: 8,
    },
    legendText: {
        color: C.text.secondary,
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.medium,
        flexShrink: 1, // Prevent long text from blowing out the bounds
    },

    /* Allocation bar row */
    allocRow: {
        flexDirection: 'row', alignItems: 'center',
        paddingVertical: 9, gap: Spacing.sm,
    },
    allocLeft: { flexDirection: 'row', alignItems: 'center', width: 90 },
    allocDot: { width: 10, height: 10, borderRadius: 5, marginRight: 6, flexShrink: 0 },
    allocName: {
        color: C.text.primary,
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
        flex: 1,
    },
    allocBarWrap: {
        flex: 1, height: 8,
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.full, overflow: 'hidden',
    },
    allocBar: { height: '100%', borderRadius: Radius.full },
    allocPct: {
        color: C.text.secondary,
        fontSize: Typography.sizes.sm,
        width: 48, textAlign: 'right',
        fontWeight: Typography.weights.semibold,
    },

    /* Disclaimer */
    disclaimer: {
        backgroundColor: C.signal.holdBg,
        borderRadius: Radius.md,
        padding: Spacing.md,
        marginTop: Spacing.sm,
        borderWidth: 1,
        borderColor: C.signal.hold + '33',
    },
    disclaimerText: { color: C.signal.hold, fontSize: Typography.sizes.xs, lineHeight: 18 },
});
