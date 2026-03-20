/**
 * Stock Detail Screen - [symbol].js
 * Shows full AI prediction, 7-day forecast chart, indicators, and explanation.
 */
import React, { useState, useRef, useMemo } from 'react';
import {
    View, Text, ScrollView, StyleSheet,
    Dimensions, TouchableOpacity, RefreshControl, Animated,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { LineChart } from 'react-native-chart-kit';
import { useStockDetail } from '../../src/hooks/useStockData';
import {
    Card, SignalBadge, RiskBadge, LoadingSkeleton, ErrorBanner,
    IndicatorRow, SectionHeader,
} from '../../src/components/ui';
import { Spacing, Radius, Typography, Shadow } from '../../src/constants/theme';
import { useColors } from '../../src/context/ThemeContext';

const W = Dimensions.get('window').width;

export default function StockDetailScreen() {
    const { symbol } = useLocalSearchParams();
    const router = useRouter();
    const { data, loading, error, refetch } = useStockDetail(symbol);
    const [refreshing, setRefreshing] = useState(false);
    const headerAnim = useRef(new Animated.Value(0)).current;
    const insets = useSafeAreaInsets();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    React.useEffect(() => {
        Animated.timing(headerAnim, {
            toValue: 1, duration: 500, useNativeDriver: true,
        }).start();
    }, []);

    const onRefresh = async () => {
        setRefreshing(true);
        await refetch();
        setRefreshing(false);
    };

    // Build 7-day chart data
    const chartData = React.useMemo(() => {
        if (!data) return null;
        const forecast = data.seven_day_forecast || [];
        const prices = [data.current_price, ...forecast].map((p) => Math.round(p));
        const labels = ['Now', ...Array.from({ length: forecast.length }, (_, i) => `D${i + 1}`)];
        return { labels, datasets: [{ data: prices }] };
    }, [data]);

    const trendUp = data && data.predicted_price > data.current_price;
    const pct = data
        ? ((data.predicted_price - data.current_price) / data.current_price) * 100
        : 0;

    return (
        <View style={[styles.root, { paddingTop: insets.top }]}>
            {/* Custom header */}
            <View style={styles.header}>
                <TouchableOpacity
                    style={styles.backBtn}
                    onPress={() => router.back()}
                    hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                >
                    <Text style={styles.backIcon}>‹</Text>
                </TouchableOpacity>
                <Text style={styles.headerTitle} numberOfLines={1}>{symbol?.replace('.NS', '')}</Text>
                <View style={{ width: 44 }} />
            </View>

            <ScrollView
                contentContainerStyle={[styles.scroll, { paddingBottom: insets.bottom + Spacing.xxl }]}
                showsVerticalScrollIndicator={false}
                refreshControl={
                    <RefreshControl
                        refreshing={refreshing}
                        onRefresh={onRefresh}
                        tintColor={C.brand.purple}
                        colors={[C.brand.purple]}
                    />
                }
            >
                {error && <ErrorBanner message={error} onRetry={refetch} />}
                {loading ? (
                    <LoadingSkeleton rows={6} />
                ) : data ? (
                    <>
                        {/* Hero price block */}
                        <Animated.View style={[styles.heroCard, { opacity: headerAnim }]}>
                            <Text style={styles.currentPriceLabel}>Current Price</Text>
                            <Text style={styles.currentPrice}>
                                ₹{(data.current_price || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                            </Text>
                            <View style={styles.heroRow}>
                                <View style={styles.heroStat}>
                                    <Text style={styles.heroStatLabel}>Predicted (1D)</Text>
                                    <Text style={styles.heroStatValue}>
                                        ₹{(data.predicted_price || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                                    </Text>
                                </View>
                                <View style={styles.heroStat}>
                                    <Text style={styles.heroStatLabel}>Expected Return</Text>
                                    <Text style={[styles.heroStatValue, { color: trendUp ? C.signal.buy : C.signal.sell }]}>
                                        {trendUp ? '▲' : '▼'} {Math.abs(pct).toFixed(2)}%
                                    </Text>
                                </View>
                            </View>
                            <View style={styles.badgeRow}>
                                <SignalBadge signal={data.signal} />
                                <View style={{ width: 8 }} />
                                <RiskBadge level={data.risk_level} />
                                <View style={{ width: 8 }} />
                                <View style={styles.confidenceChip}>
                                    <Text style={styles.confidenceText}>
                                        🎯 {Math.round((data.signal_confidence || 0) * 100)}% confidence
                                    </Text>
                                </View>
                            </View>
                            {data.confidence_interval && (
                                <Text style={styles.ciText}>
                                    95% CI: ₹{data.confidence_interval[0].toFixed(2)} – ₹{data.confidence_interval[1].toFixed(2)}
                                </Text>
                            )}
                        </Animated.View>

                        {/* 7-Day Forecast Chart */}
                        {chartData && chartData.datasets[0].data.length > 1 && (
                            <Card style={styles.chartCard}>
                                <SectionHeader title="7-Day Price Forecast" />
                                <LineChart
                                    data={chartData}
                                    width={W - Spacing.lg * 2 - Spacing.md * 2}
                                    height={200}
                                    chartConfig={{
                                        backgroundGradientFrom: C.bg.card,
                                        backgroundGradientTo: C.bg.card,
                                        backgroundGradientFromOpacity: 0,
                                        backgroundGradientToOpacity: 0,
                                        color: (opacity = 1) => `rgba(123, 97, 255, ${opacity})`,
                                        labelColor: () => C.text.secondary,
                                        strokeWidth: 2.5,
                                        propsForDots: {
                                            r: '4',
                                            strokeWidth: '2',
                                            stroke: C.brand.purple,
                                            fill: C.bg.card,
                                        },
                                        decimalPlaces: 0,
                                        propsForLabels: { fontSize: 10 },
                                    }}
                                    bezier
                                    style={{ borderRadius: Radius.md }}
                                    withInnerLines={false}
                                    withOuterLines={false}
                                />
                            </Card>
                        )}

                        {/* AI Explanation */}
                        {data.explanation && (
                            <Card style={styles.explainCard}>
                                <View style={styles.explainHeader}>
                                    <Text style={styles.explainIcon}>🤖</Text>
                                    <Text style={styles.explainTitle}>AI Analysis</Text>
                                </View>
                                <Text style={styles.explainText}>{data.explanation}</Text>
                            </Card>
                        )}

                        {/* Technical Indicators */}
                        {data.indicators && (
                            <Card>
                                <SectionHeader title="Technical Indicators" />
                                <IndicatorRow label="RSI (14)" value={data.indicators.rsi} highlight={
                                    data.indicators.rsi > 70 || data.indicators.rsi < 30
                                } />
                                <IndicatorRow label="MACD" value={data.indicators.macd} />
                                <IndicatorRow label="MACD Signal" value={data.indicators.macd_signal} />
                                <IndicatorRow label="SMA 20" value={data.indicators.sma_20} />
                                <IndicatorRow label="SMA 50" value={data.indicators.sma_50} />
                                <IndicatorRow label="Bollinger High" value={data.indicators.bb_high} />
                                <IndicatorRow label="Bollinger Low" value={data.indicators.bb_low} />
                                <IndicatorRow label="VWAP" value={data.indicators.vwap} />
                                <IndicatorRow label="ATR (Volatility)" value={data.indicators.atr} />
                            </Card>
                        )}

                        {/* RSI Meter */}
                        {data.indicators?.rsi && (
                            <Card>
                                <SectionHeader title="RSI Meter" />
                                <View style={styles.rsiBarBg}>
                                    <View style={[styles.rsiBarFill, { width: `${Math.min(data.indicators.rsi, 100)}%` }]} />
                                    <View style={[styles.rsiMarker, { left: '30%' }]} />
                                    <View style={[styles.rsiMarker, { left: '70%' }]} />
                                </View>
                                <View style={styles.rsiLabels}>
                                    <Text style={[styles.rsiLabel, { color: C.signal.sell }]}>Oversold</Text>
                                    <Text style={styles.rsiCurrent}>{data.indicators.rsi.toFixed(1)}</Text>
                                    <Text style={[styles.rsiLabel, { color: C.signal.buy }]}>Overbought</Text>
                                </View>
                            </Card>
                        )}
                    </>
                ) : null}
            </ScrollView>
        </View>
    );
}

const makeStyles = (C) => StyleSheet.create({
    root: { flex: 1, backgroundColor: C.bg.primary },
    header: {
        flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
        paddingHorizontal: Spacing.lg, paddingVertical: Spacing.md,
        borderBottomWidth: 1, borderBottomColor: C.border.default,
    },
    backBtn: {
        width: 44, height: 44, alignItems: 'center', justifyContent: 'center',
        backgroundColor: C.bg.elevated, borderRadius: Radius.md,
        borderWidth: 1, borderColor: C.border.default,
    },
    backIcon: { color: C.text.primary, fontSize: 26, fontWeight: Typography.weights.bold, lineHeight: 30 },
    headerTitle: {
        color: C.text.primary, fontSize: Typography.sizes.xl,
        fontWeight: Typography.weights.bold, letterSpacing: 1,
    },
    scroll: { padding: Spacing.lg, paddingBottom: Spacing.xxl },
    heroCard: {
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.xl,
        padding: Spacing.lg,
        marginBottom: Spacing.md,
        borderWidth: 1, borderColor: C.border.brand,
        ...Shadow.glow,
    },
    currentPriceLabel: { color: C.text.muted, fontSize: Typography.sizes.sm },
    currentPrice: {
        color: C.text.primary, fontSize: Typography.sizes.display,
        fontWeight: Typography.weights.black, marginVertical: 4,
    },
    heroRow: { flexDirection: 'row', gap: Spacing.xl, marginBottom: Spacing.md, flexWrap: 'wrap' },
    heroStat: { minWidth: 120 },
    heroStatLabel: { color: C.text.muted, fontSize: Typography.sizes.xs },
    heroStatValue: { color: C.text.primary, fontSize: Typography.sizes.lg, fontWeight: Typography.weights.bold, marginTop: 2 },
    badgeRow: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 4 },
    confidenceChip: {
        backgroundColor: C.brand.glow,
        paddingHorizontal: 10, paddingVertical: 4, borderRadius: Radius.full,
    },
    confidenceText: { color: C.brand.purple, fontSize: Typography.sizes.xs, fontWeight: Typography.weights.semibold },
    ciText: { color: C.text.muted, fontSize: Typography.sizes.xs, marginTop: Spacing.sm },
    chartCard: { marginBottom: Spacing.md },
    explainCard: { marginBottom: Spacing.md },
    explainHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: Spacing.sm },
    explainIcon: { fontSize: 20, marginRight: 6 },
    explainTitle: { color: C.brand.purpleLight, fontSize: Typography.sizes.md, fontWeight: Typography.weights.bold },
    explainText: { color: C.text.secondary, fontSize: Typography.sizes.sm, lineHeight: 20 },
    rsiBarBg: {
        height: 12, backgroundColor: C.bg.elevated, borderRadius: Radius.full,
        overflow: 'hidden', marginTop: Spacing.sm, position: 'relative',
    },
    rsiBarFill: { height: '100%', backgroundColor: C.brand.purple, borderRadius: Radius.full },
    rsiMarker: { position: 'absolute', top: 0, width: 1, height: '100%', backgroundColor: C.text.muted },
    rsiLabels: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 6 },
    rsiLabel: { fontSize: Typography.sizes.xs },
    rsiCurrent: { color: C.text.primary, fontWeight: Typography.weights.bold, fontSize: Typography.sizes.sm },
});
