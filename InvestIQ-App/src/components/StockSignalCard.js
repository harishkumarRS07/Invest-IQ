/**
 * StockSignalCard - Phase 3 Enhanced Signal Card
 * Shows ticker, price, signal, confidence, indicators, and more.
 * 
 * Props:
 * - item: { symbol, signal, signal_confidence, confidence, current_price, 
 *           risk_level, indicators, probabilities, explanation, pct_change }
 * - index: For animation stagger
 * - onPress: Optional custom press handler
 */
import React, { useRef, useEffect, useMemo } from 'react';
import {
    View, Text, TouchableOpacity, StyleSheet, Animated, ScrollView,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SignalBadge, RiskBadge } from './ui';
import { Spacing, Radius, Typography, Shadow } from '../constants/theme';
import { useColors } from '../context/ThemeContext';

export default function StockSignalCard({ item, index = 0, onPress, showDetails = false }) {
    const router = useRouter();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    const fadeAnim = useRef(new Animated.Value(0)).current;
    const slideAnim = useRef(new Animated.Value(20)).current;

    useEffect(() => {
        Animated.parallel([
            Animated.timing(fadeAnim, {
                toValue: 1, duration: 350,
                delay: index * 70, useNativeDriver: true,
            }),
            Animated.timing(slideAnim, {
                toValue: 0, duration: 350,
                delay: index * 70, useNativeDriver: true,
            }),
        ]).start();
    }, []);

    // Get confidence (try both field names)
    const confidence = item.signal_confidence || item.confidence || 0.5;
    const confidencePercent = Math.round(confidence * 100);

    const pct = typeof item.pct_change === 'number' ? item.pct_change : 0;
    const isPositive = pct >= 0;
    const changeColor = isPositive ? C.signal.buy : C.signal.sell;
    const symbol = (item.symbol || '').replace('.NS', '');

    // Get signal color
    const getSignalBg = () => {
        if (item.signal === 'BUY') return C.signal.buy;
        if (item.signal === 'SELL') return C.signal.sell;
        return C.signal.hold;
    };

    const handlePress = () => {
        if (onPress) {
            onPress(item);
        } else {
            router.push(`/stock/${item.symbol}`);
        }
    };

    return (
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
            <TouchableOpacity
                activeOpacity={0.85}
                style={[styles.card, { borderLeftColor: getSignalBg(), borderLeftWidth: 4 }]}
                onPress={handlePress}
            >
                {/* Top row: Symbol + Price */}
                <View style={styles.topRow}>
                    <View style={styles.left}>
                        <View style={[styles.symbolChip, { backgroundColor: getSignalBg() + '20' }]}>
                            <Text style={[styles.symbolText, { color: getSignalBg() }]} numberOfLines={1}>
                                {symbol.slice(0, 4)}
                            </Text>
                        </View>
                        <View style={styles.nameBlock}>
                            <Text style={styles.symbolLabel} numberOfLines={1}>{symbol}</Text>
                            <RiskBadge level={item.risk_level || 'Medium'} />
                        </View>
                    </View>
                    <View style={styles.right}>
                        <Text style={styles.price} numberOfLines={1}>
                            ₹{(item.current_price || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                        </Text>
                        <Text style={[styles.change, { color: changeColor }]}>
                            {isPositive ? '▲' : '▼'} {Math.abs(pct).toFixed(2)}%
                        </Text>
                    </View>
                </View>

                {/* Signal + Confidence Section */}
                <View style={styles.signalSection}>
                    <View style={styles.signalRow}>
                        <SignalBadge signal={item.signal} />
                        <Text style={[styles.confidenceText, { color: getSignalBg() }]}>
                            {confidencePercent}% confidence
                        </Text>
                    </View>

                    {/* Confidence Bar */}
                    <View style={styles.confidenceBarBg}>
                        <View
                            style={[
                                styles.confidenceBarFill,
                                { width: `${confidencePercent}%`, backgroundColor: getSignalBg() },
                            ]}
                        />
                    </View>
                </View>

                {/* Probabilities (if available) */}
                {item.probabilities && (
                    <View style={styles.probabilitiesRow}>
                        <View style={styles.probItem}>
                            <Text style={styles.probLabel}>BUY</Text>
                            <Text style={[styles.probValue, { color: C.signal.buy }]}>
                                {Math.round((item.probabilities.buy || 0) * 100)}%
                            </Text>
                        </View>
                        <View style={styles.probItem}>
                            <Text style={styles.probLabel}>HOLD</Text>
                            <Text style={[styles.probValue, { color: C.signal.hold }]}>
                                {Math.round((item.probabilities.hold || 0) * 100)}%
                            </Text>
                        </View>
                        <View style={styles.probItem}>
                            <Text style={styles.probLabel}>SELL</Text>
                            <Text style={[styles.probValue, { color: C.signal.sell }]}>
                                {Math.round((item.probabilities.sell || 0) * 100)}%
                            </Text>
                        </View>
                    </View>
                )}

                {/* Key Indicators (if showing details) */}
                {showDetails && item.indicators && (
                    <View style={styles.indicatorsRow}>
                        {item.indicators.rsi !== null && (
                            <View style={styles.indicatorBadge}>
                                <Text style={styles.indicatorLabel}>RSI</Text>
                                <Text style={styles.indicatorValue}>
                                    {parseFloat(item.indicators.rsi).toFixed(1)}
                                </Text>
                            </View>
                        )}
                        {item.indicators.macd !== null && (
                            <View style={styles.indicatorBadge}>
                                <Text style={styles.indicatorLabel}>MACD</Text>
                                <Text style={styles.indicatorValue}>
                                    {parseFloat(item.indicators.macd).toFixed(2)}
                                </Text>
                            </View>
                        )}
                        {item.indicators.sma_20 !== null && (
                            <View style={styles.indicatorBadge}>
                                <Text style={styles.indicatorLabel}>SMA20</Text>
                                <Text style={styles.indicatorValue}>
                                    {parseFloat(item.indicators.sma_20).toFixed(0)}
                                </Text>
                            </View>
                        )}
                    </View>
                )}

                {/* Explanation (if available and showing details) */}
                {showDetails && item.explanation && (
                    <View style={styles.explanationBox}>
                        <Text style={styles.explanationText} numberOfLines={2}>
                            {item.explanation}
                        </Text>
                    </View>
                )}
            </TouchableOpacity>
        </Animated.View>
    );
}

const makeStyles = (C) => StyleSheet.create({
    card: {
        backgroundColor: C.bg.card,
        borderRadius: Radius.lg,
        padding: Spacing.md,
        marginBottom: Spacing.sm,
        borderWidth: 1,
        borderColor: C.border.default,
        ...Shadow.card,
    },

    // Top row: symbol + price
    topRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: Spacing.sm,
    },
    left: { 
        flexDirection: 'row', 
        alignItems: 'center', 
        flex: 1, 
        marginRight: Spacing.sm 
    },
    symbolChip: {
        width: 46, 
        height: 46, 
        borderRadius: Radius.md,
        borderWidth: 2,
        alignItems: 'center', 
        justifyContent: 'center', 
        flexShrink: 0,
    },
    symbolText: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.black,
        letterSpacing: 0.5,
    },
    nameBlock: { 
        marginLeft: Spacing.sm, 
        flexShrink: 1 
    },
    symbolLabel: {
        color: C.text.primary,
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.bold,
        marginBottom: 3,
    },
    right: { 
        alignItems: 'flex-end', 
        flexShrink: 0 
    },
    price: { 
        color: C.text.primary, 
        fontSize: Typography.sizes.lg, 
        fontWeight: Typography.weights.bold 
    },
    change: { 
        fontSize: Typography.sizes.sm, 
        fontWeight: Typography.weights.semibold, 
        marginTop: 2 
    },

    // Signal + Confidence Section
    signalSection: {
        marginBottom: Spacing.sm,
        paddingBottom: Spacing.sm,
        borderBottomWidth: 1,
        borderBottomColor: C.border.default,
    },
    signalRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: Spacing.xs,
    },
    confidenceText: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
        letterSpacing: 0.3,
    },

    // Confidence Bar
    confidenceBarBg: {
        height: 6,
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.full,
        overflow: 'hidden',
    },
    confidenceBarFill: {
        height: 6,
        borderRadius: Radius.full,
    },

    // Probabilities Row
    probabilitiesRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        marginBottom: Spacing.sm,
        paddingBottom: Spacing.sm,
        borderBottomWidth: 1,
        borderBottomColor: C.border.default,
    },
    probItem: {
        alignItems: 'center',
        flex: 1,
    },
    probLabel: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
        color: C.text.muted,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
        marginBottom: 2,
    },
    probValue: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.bold,
    },

    // Indicators Row
    indicatorsRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        marginBottom: Spacing.sm,
        paddingBottom: Spacing.sm,
        borderBottomWidth: 1,
        borderBottomColor: C.border.default,
    },
    indicatorBadge: {
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.md,
        paddingHorizontal: Spacing.sm,
        paddingVertical: Spacing.xs,
        alignItems: 'center',
        flex: 1,
        marginHorizontal: 2,
    },
    indicatorLabel: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
        color: C.text.muted,
        textTransform: 'uppercase',
        letterSpacing: 0.4,
    },
    indicatorValue: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.bold,
        color: C.text.primary,
        marginTop: 2,
    },

    // Explanation Box
    explanationBox: {
        backgroundColor: C.bg.elevated,
        borderRadius: Radius.md,
        borderLeftWidth: 3,
        borderLeftColor: C.brand.blue,
        paddingHorizontal: Spacing.sm,
        paddingVertical: Spacing.xs,
    },
    explanationText: {
        fontSize: Typography.sizes.xs,
        color: C.text.secondary,
        lineHeight: 16,
        fontWeight: Typography.weights.medium,
    },
});
