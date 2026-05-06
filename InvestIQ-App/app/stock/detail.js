/**
 * Stock Detail Page - app/stock/[symbol].js
 * Full stock analysis with signals, indicators, predictions, and more
 * 
 * This page is loaded when user taps on a stock card
 * Shows comprehensive trading signal and technical analysis
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    TouchableOpacity,
    ActivityIndicator,
    Alert,
    SafeAreaView,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import api from '../../src/services/api';
import { useColors } from '../../src/context/ThemeContext';
import { Spacing, Radius, Typography, Shadow } from '../../src/constants/theme';

export default function StockDetailPage() {
    const router = useRouter();
    const { symbol } = useLocalSearchParams();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    // State management
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('signal'); // 'signal', 'indicators', 'probabilities'

    // Fetch prediction data
    const fetchPrediction = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);

            // Call API to get prediction
            const response = await api.post('/predict', {
                symbol: symbol.replace('.NS', ''),
                model: 'transformer',
            });

            setData(response.data);
        } catch (err) {
            console.error('Error fetching prediction:', err);
            setError(err.message || 'Failed to load prediction');
        } finally {
            setLoading(false);
        }
    }, [symbol]);

    // Initial load
    useEffect(() => {
        if (symbol) {
            fetchPrediction();
        }
    }, [symbol, fetchPrediction]);

    // Utility function: get signal color
    const getSignalColor = (signal) => {
        if (signal === 'BUY') return { bg: '#dcfce7', text: '#22c55e', dark: '#16a34a' };
        if (signal === 'SELL') return { bg: '#fee2e2', text: '#ef4444', dark: '#dc2626' };
        return { bg: '#fef3c7', text: '#f59e0b', dark: '#d97706' };
    };

    // Utility function: format numbers
    const formatNumber = (num, decimals = 2) => {
        if (num === null || num === undefined) return 'N/A';
        return parseFloat(num).toFixed(decimals);
    };

    // Loading state
    if (loading) {
        return (
            <SafeAreaView style={[styles.container, { backgroundColor: C.bg.primary }]}>
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => router.back()}>
                        <Text style={styles.backButton}>← Back</Text>
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>{symbol}</Text>
                    <View style={{ width: 40 }} />
                </View>

                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color={C.brand.blue} />
                    <Text style={[styles.loadingText, { color: C.text.secondary }]}>
                        Loading analysis...
                    </Text>
                </View>
            </SafeAreaView>
        );
    }

    // Error state
    if (error || !data) {
        return (
            <SafeAreaView style={[styles.container, { backgroundColor: C.bg.primary }]}>
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => router.back()}>
                        <Text style={styles.backButton}>← Back</Text>
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>{symbol}</Text>
                    <View style={{ width: 40 }} />
                </View>

                <View style={styles.errorContainer}>
                    <Text style={[styles.errorText, { color: C.text.primary }]}>Color: {error || 'No data available'}</Text>
                    <TouchableOpacity style={styles.retryButton} onPress={fetchPrediction}>
                        <Text style={styles.retryButtonText}>Try Again</Text>
                    </TouchableOpacity>
                </View>
            </SafeAreaView>
        );
    }

    const signal = data.signal || 'HOLD';
    const confidence = data.signal_confidence || data.confidence || 0;
    const confidencePercent = Math.round(confidence * 100);
    const signalColors = getSignalColor(signal);
    const indicators = data.indicators || {};
    const probabilities = data.probabilities || {};

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: C.bg.primary }]}>
            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <Text style={styles.backButton}>← Back</Text>
                </TouchableOpacity>
                <Text style={styles.headerTitle}>{symbol}</Text>
                <TouchableOpacity onPress={fetchPrediction}>
                    <Text style={styles.refreshButton}>⟳</Text>
                </TouchableOpacity>
            </View>

            <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
                {/* CURRENT PRICE SECTION */}
                {data.current_price && (
                    <View style={[styles.section, styles.priceSection]}>
                        <Text style={[styles.priceLabel, { color: C.text.muted }]}>Current Price</Text>
                        <Text style={[styles.priceValue, { color: C.text.primary }]}>
                            ₹{formatNumber(data.current_price, 2)}
                        </Text>
                    </View>
                )}

                {/* MAIN SIGNAL CARD */}
                <View style={[styles.section, { backgroundColor: signalColors.bg, borderLeftColor: signalColors.text, borderLeftWidth: 4 }]}>
                    <View style={styles.signalHeader}>
                        <View>
                            <Text style={[styles.sectionLabel, { color: C.text.muted }]}>TRADING SIGNAL</Text>
                            <Text style={[styles.signalBig, { color: signalColors.dark }]}>
                                {signal}
                            </Text>
                        </View>
                        <Text style={[styles.confidencePercentBig, { color: signalColors.dark }]}>
                            {confidencePercent}%
                        </Text>
                    </View>

                    {/* Confidence Bar */}
                    <View style={styles.confidenceContainer}>
                        <View style={styles.confidenceBarBackground}>
                            <View
                                style={[
                                    styles.confidenceBarFill,
                                    {
                                        width: `${confidencePercent}%`,
                                        backgroundColor: signalColors.text,
                                    },
                                ]}
                            />
                        </View>
                    </View>

                    {/* Signal Description */}
                    {data.explanation && (
                        <Text style={[styles.explanation, { color: signalColors.dark, marginTop: Spacing.sm }]}>
                            {data.explanation}
                        </Text>
                    )}
                </View>

                {/* TABS */}
                <View style={styles.tabsContainer}>
                    <TouchableOpacity
                        style={[styles.tab, activeTab === 'signal' && styles.tabActive]}
                        onPress={() => setActiveTab('signal')}
                    >
                        <Text style={[styles.tabText, activeTab === 'signal' && styles.tabTextActive]}>
                            Probabilities
                        </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={[styles.tab, activeTab === 'indicators' && styles.tabActive]}
                        onPress={() => setActiveTab('indicators')}
                    >
                        <Text style={[styles.tabText, activeTab === 'indicators' && styles.tabTextActive]}>
                            Indicators
                        </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={[styles.tab, activeTab === 'probabilities' && styles.tabActive]}
                        onPress={() => setActiveTab('probabilities')}
                    >
                        <Text style={[styles.tabText, activeTab === 'probabilities' && styles.tabTextActive]}>
                            Forecast
                        </Text>
                    </TouchableOpacity>
                </View>

                {/* PROBABILITIES TAB */}
                {activeTab === 'signal' && probabilities && (
                    <View style={styles.section}>
                        <Text style={[styles.sectionTitle, { color: C.text.primary }]}>Signal Probabilities</Text>

                        <View style={styles.probabilityItem}>
                            <View style={styles.probabilityLeft}>
                                <Text style={[styles.probabilityLabel, { color: '#22c55e' }]}>BUY</Text>
                                <Text style={[styles.probabilityDescription]}>Stock predicted to rise</Text>
                            </View>
                            <Text style={[styles.probabilityValue, { color: '#22c55e' }]}>
                                {formatNumber(probabilities.buy * 100, 1)}%
                            </Text>
                        </View>

                        <View style={styles.probabilityItem}>
                            <View style={styles.probabilityLeft}>
                                <Text style={[styles.probabilityLabel, { color: '#f59e0b' }]}>HOLD</Text>
                                <Text style={[styles.probabilityDescription]}>No clear direction</Text>
                            </View>
                            <Text style={[styles.probabilityValue, { color: '#f59e0b' }]}>
                                {formatNumber(probabilities.hold * 100, 1)}%
                            </Text>
                        </View>

                        <View style={styles.probabilityItem}>
                            <View style={styles.probabilityLeft}>
                                <Text style={[styles.probabilityLabel, { color: '#ef4444' }]}>SELL</Text>
                                <Text style={[styles.probabilityDescription]}>Stock predicted to fall</Text>
                            </View>
                            <Text style={[styles.probabilityValue, { color: '#ef4444' }]}>
                                {formatNumber(probabilities.sell * 100, 1)}%
                            </Text>
                        </View>
                    </View>
                )}

                {/* INDICATORS TAB */}
                {activeTab === 'indicators' && indicators && (
                    <View style={styles.section}>
                        <Text style={[styles.sectionTitle, { color: C.text.primary }]}>Technical Indicators</Text>

                        <View style={styles.indicatorGrid}>
                            {indicators.rsi !== null && (
                                <View style={styles.indicatorBox}>
                                    <Text style={[styles.indicatorBoxLabel, { color: C.text.muted }]}>RSI</Text>
                                    <Text style={[styles.indicatorBoxValue, { color: C.text.primary }]}>
                                        {formatNumber(indicators.rsi, 2)}
                                    </Text>
                                    <Text style={[styles.indicatorBoxDescription, { color: C.text.secondary }]}>
                                        {indicators.rsi > 70 ? '🔴 Overbought' : indicators.rsi < 30 ? '🟢 Oversold' : '🟡 Neutral'}
                                    </Text>
                                </View>
                            )}

                            {indicators.macd !== null && (
                                <View style={styles.indicatorBox}>
                                    <Text style={[styles.indicatorBoxLabel, { color: C.text.muted }]}>MACD</Text>
                                    <Text style={[styles.indicatorBoxValue, { color: C.text.primary }]}>
                                        {formatNumber(indicators.macd, 4)}
                                    </Text>
                                    <Text style={[styles.indicatorBoxDescription, { color: C.text.secondary }]}>
                                        {indicators.macd > 0 ? '🟢 Bullish' : '🔴 Bearish'}
                                    </Text>
                                </View>
                            )}

                            {indicators.sma_20 !== null && (
                                <View style={styles.indicatorBox}>
                                    <Text style={[styles.indicatorBoxLabel, { color: C.text.muted }]}>SMA 20</Text>
                                    <Text style={[styles.indicatorBoxValue, { color: C.text.primary }]}>
                                        ₹{formatNumber(indicators.sma_20, 2)}
                                    </Text>
                                    <Text style={[styles.indicatorBoxDescription, { color: C.text.secondary }]}>
                                        Short-term trend
                                    </Text>
                                </View>
                            )}

                            {indicators.sma_50 !== null && (
                                <View style={styles.indicatorBox}>
                                    <Text style={[styles.indicatorBoxLabel, { color: C.text.muted }]}>SMA 50</Text>
                                    <Text style={[styles.indicatorBoxValue, { color: C.text.primary }]}>
                                        ₹{formatNumber(indicators.sma_50, 2)}
                                    </Text>
                                    <Text style={[styles.indicatorBoxDescription, { color: C.text.secondary }]}>
                                        Long-term trend
                                    </Text>
                                </View>
                            )}

                            {indicators.atr !== null && (
                                <View style={styles.indicatorBox}>
                                    <Text style={[styles.indicatorBoxLabel, { color: C.text.muted }]}>ATR</Text>
                                    <Text style={[styles.indicatorBoxValue, { color: C.text.primary }]}>
                                        {formatNumber(indicators.atr, 2)}
                                    </Text>
                                    <Text style={[styles.indicatorBoxDescription, { color: C.text.secondary }]}>
                                        Volatility measure
                                    </Text>
                                </View>
                            )}

                            {indicators.bb_high !== null && indicators.bb_low !== null && (
                                <View style={styles.indicatorBox}>
                                    <Text style={[styles.indicatorBoxLabel, { color: C.text.muted }]}>BB Range</Text>
                                    <Text style={[styles.indicatorBoxValue, { color: C.text.primary }]}>
                                        {formatNumber(indicators.bb_high - indicators.bb_low, 2)}
                                    </Text>
                                    <Text style={[styles.indicatorBoxDescription, { color: C.text.secondary }]}>
                                        Band width
                                    </Text>
                                </View>
                            )}
                        </View>
                    </View>
                )}

                {/* FORECAST TAB */}
                {activeTab === 'probabilities' && data.seven_day_forecast && (
                    <View style={styles.section}>
                        <Text style={[styles.sectionTitle, { color: C.text.primary }]}>7-Day Forecast</Text>

                        {data.seven_day_forecast.map((prediction, index) => (
                            <View key={index} style={styles.forecastRow}>
                                <Text style={[styles.forecastDay, { color: C.text.primary }]}>
                                    Day {index + 1}
                                </Text>
                                <View style={styles.forecastBar}>
                                    <View style={[
                                        styles.forecastBarFill,
                                        {
                                            width: `${Math.abs(prediction) * 1000 > 100 ? 100 : Math.abs(prediction) * 1000}%`,
                                            backgroundColor: prediction > 0 ? '#22c55e' : '#ef4444',
                                        }
                                    ]} />
                                </View>
                                <Text style={[
                                    styles.forecastValue,
                                    { color: prediction > 0 ? '#22c55e' : '#ef4444' }
                                ]}>
                                    {prediction > 0 ? '+' : ''}{formatNumber(prediction * 100, 2)}%
                                </Text>
                            </View>
                        ))}
                    </View>
                )}

                {/* RISK & SENTIMENT */}
                {(data.risk_level || data.sentiment) && (
                    <View style={styles.section}>
                        <Text style={[styles.sectionTitle, { color: C.text.primary }]}>Risk & Sentiment</Text>

                        <View style={styles.metadataRow}>
                            {data.risk_level && (
                                <View style={[styles.metadataBox, { borderColor: C.border.default }]}>
                                    <Text style={[styles.metadataLabel, { color: C.text.muted }]}>Risk Level</Text>
                                    <Text style={[styles.metadataValue, {
                                        color: data.risk_level === 'high' ? '#ef4444'
                                            : data.risk_level === 'medium' ? '#f59e0b'
                                                : '#22c55e'
                                    }]}>
                                        {data.risk_level.toUpperCase()}
                                    </Text>
                                </View>
                            )}

                            {data.sentiment && (
                                <View style={[styles.metadataBox, { borderColor: C.border.default }]}>
                                    <Text style={[styles.metadataLabel, { color: C.text.muted }]}>Sentiment</Text>
                                    <Text style={[styles.metadataValue, {
                                        color: data.sentiment === 'bullish' ? '#22c55e'
                                            : data.sentiment === 'bearish' ? '#ef4444'
                                                : '#f59e0b'
                                    }]}>
                                        {data.sentiment.toUpperCase()}
                                    </Text>
                                </View>
                            )}
                        </View>
                    </View>
                )}

                {/* MODEL INFO */}
                <View style={[styles.section, { backgroundColor: C.bg.elevated }]}>
                    <Text style={[styles.modelLabel, { color: C.text.muted }]}>Model Information</Text>
                    <Text style={[styles.modelValue, { color: C.text.primary }]}>
                        🧠 Transformer Neural Network (Phase 3)
                    </Text>
                    <Text style={[styles.modelDescription, { color: C.text.secondary }]}>
                        Advanced deep learning model trained on 25 years of historical data with technical indicators and market correlation.
                    </Text>
                </View>

                {/* FOOTER */}
                <View style={{ height: 30 }} />
            </ScrollView>
        </SafeAreaView>
    );
}

const makeStyles = (C) => StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: C.bg.primary,
    },

    // HEADER
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: Spacing.md,
        paddingVertical: Spacing.sm,
        backgroundColor: C.bg.card,
        borderBottomWidth: 1,
        borderBottomColor: C.border.default,
    },
    backButton: {
        fontSize: Typography.sizes.md,
        color: C.brand.blue,
        fontWeight: Typography.weights.bold,
    },
    headerTitle: {
        fontSize: Typography.sizes.lg,
        fontWeight: Typography.weights.bold,
        color: C.text.primary,
    },
    refreshButton: {
        fontSize: Typography.sizes.lg,
        color: C.brand.blue,
        fontWeight: Typography.weights.bold,
    },

    // LOADING & ERROR
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    loadingText: {
        marginTop: Spacing.md,
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.medium,
    },
    errorContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: Spacing.lg,
    },
    errorText: {
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.medium,
        textAlign: 'center',
        marginBottom: Spacing.md,
    },
    retryButton: {
        backgroundColor: C.brand.blue,
        paddingHorizontal: Spacing.lg,
        paddingVertical: Spacing.md,
        borderRadius: Radius.md,
    },
    retryButtonText: {
        color: '#fff',
        fontWeight: Typography.weights.bold,
        fontSize: Typography.sizes.md,
    },

    // CONTENT
    content: {
        flex: 1,
        paddingHorizontal: Spacing.md,
        paddingTop: Spacing.sm,
    },

    // SECTIONS
    section: {
        backgroundColor: C.bg.card,
        borderRadius: Radius.lg,
        padding: Spacing.md,
        marginBottom: Spacing.md,
        borderWidth: 1,
        borderColor: C.border.default,
    },
    priceSection: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: Spacing.lg,
    },
    priceLabel: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    priceValue: {
        fontSize: 32,
        fontWeight: Typography.weights.black,
        marginTop: Spacing.xs,
    },

    // SIGNAL SECTION
    signalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.md,
    },
    sectionLabel: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
        textTransform: 'uppercase',
        letterSpacing: 1,
    },
    signalBig: {
        fontSize: 32,
        fontWeight: Typography.weights.black,
        marginTop: Spacing.xs,
    },
    confidencePercentBig: {
        fontSize: 28,
        fontWeight: Typography.weights.black,
    },
    confidenceContainer: {
        marginBottom: Spacing.md,
    },
    confidenceBarBackground: {
        height: 8,
        backgroundColor: 'rgba(0,0,0,0.1)',
        borderRadius: Radius.full,
        overflow: 'hidden',
    },
    confidenceBarFill: {
        height: 8,
        borderRadius: Radius.full,
    },
    explanation: {
        fontSize: Typography.sizes.sm,
        lineHeight: 18,
        fontWeight: Typography.weights.medium,
    },

    // TABS
    tabsContainer: {
        flexDirection: 'row',
        marginBottom: Spacing.md,
        backgroundColor: C.bg.card,
        borderRadius: Radius.lg,
        paddingHorizontal: Spacing.xs,
        paddingVertical: Spacing.xs,
        borderWidth: 1,
        borderColor: C.border.default,
    },
    tab: {
        flex: 1,
        paddingVertical: Spacing.sm,
        paddingHorizontal: Spacing.sm,
        borderRadius: Radius.md,
        alignItems: 'center',
    },
    tabActive: {
        backgroundColor: C.brand.blue,
    },
    tabText: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
        color: C.text.secondary,
    },
    tabTextActive: {
        color: '#fff',
    },

    // SECTION TITLE
    sectionTitle: {
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.bold,
        marginBottom: Spacing.md,
    },

    // PROBABILITY ITEMS
    probabilityItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: Spacing.sm,
        borderBottomWidth: 1,
        borderBottomColor: C.border.default,
    },
    probabilityLeft: {
        flex: 1,
    },
    probabilityLabel: {
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.bold,
        marginBottom: Spacing.xs,
    },
    probabilityDescription: {
        fontSize: Typography.sizes.xs,
        color: C.text.secondary,
    },
    probabilityValue: {
        fontSize: Typography.sizes.lg,
        fontWeight: Typography.weights.bold,
    },

    // INDICATOR GRID
    indicatorGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
    },
    indicatorBox: {
        width: '48%',
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.md,
        padding: Spacing.sm,
        marginBottom: Spacing.md,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: C.border.default,
    },
    indicatorBoxLabel: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
        marginBottom: Spacing.xs,
    },
    indicatorBoxValue: {
        fontSize: Typography.sizes.lg,
        fontWeight: Typography.weights.bold,
        marginBottom: Spacing.xs,
    },
    indicatorBoxDescription: {
        fontSize: Typography.sizes.xs,
        textAlign: 'center',
    },

    // FORECAST
    forecastRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: Spacing.sm,
        paddingVertical: Spacing.xs,
    },
    forecastDay: {
        width: 50,
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.bold,
    },
    forecastBar: {
        flex: 1,
        height: 6,
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.full,
        overflow: 'hidden',
        marginHorizontal: Spacing.sm,
    },
    forecastBarFill: {
        height: 6,
        borderRadius: Radius.full,
    },
    forecastValue: {
        width: 60,
        textAlign: 'right',
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.bold,
    },

    // METADATA
    metadataRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
    },
    metadataBox: {
        flex: 1,
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.md,
        padding: Spacing.md,
        borderWidth: 1,
        marginHorizontal: Spacing.xs,
        alignItems: 'center',
    },
    metadataLabel: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
        marginBottom: Spacing.xs,
    },
    metadataValue: {
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.bold,
    },

    // MODEL INFO
    modelLabel: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
        marginBottom: Spacing.xs,
    },
    modelValue: {
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.bold,
        marginBottom: Spacing.xs,
    },
    modelDescription: {
        fontSize: Typography.sizes.xs,
        lineHeight: 16,
    },
});
