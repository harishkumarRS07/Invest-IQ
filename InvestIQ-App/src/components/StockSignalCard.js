/**
 * StockSignalCard - Premium card for the Dashboard FlatList.
 * Shows ticker, price, predicted price, pct change, signal, and risk.
 */
import React, { useRef, useEffect, useMemo } from 'react';
import {
    View, Text, TouchableOpacity, StyleSheet, Animated,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SignalBadge, RiskBadge } from './ui';
import { Spacing, Radius, Typography, Shadow } from '../constants/theme';
import { useColors } from '../context/ThemeContext';

export default function StockSignalCard({ item, index = 0 }) {
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

    const pct = typeof item.pct_change === 'number' ? item.pct_change : 0;
    const isPositive = pct >= 0;
    const changeColor = isPositive ? C.signal.buy : C.signal.sell;
    const symbol = (item.symbol || '').replace('.NS', '');

    return (
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
            <TouchableOpacity
                activeOpacity={0.85}
                style={styles.card}
                onPress={() => router.push(`/stock/${item.symbol}`)}
            >
                {/* Top row */}
                <View style={styles.topRow}>
                    <View style={styles.left}>
                        <View style={styles.symbolChip}>
                            <Text style={styles.symbolText} numberOfLines={1}>
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

                {/* Bottom row: confidence bar + signal badge */}
                <View style={styles.bottomRow}>
                    <View style={styles.confidenceWrap}>
                        <View style={styles.confidenceBarBg}>
                            <View
                                style={[
                                    styles.confidenceBarFill,
                                    {
                                        width: `${Math.round((item.signal_confidence || 0.5) * 100)}%`,
                                        backgroundColor: item.signal === 'BUY' ? C.signal.buy
                                            : item.signal === 'SELL' ? C.signal.sell
                                                : C.signal.hold,
                                    },
                                ]}
                            />
                        </View>
                        <Text style={styles.confidenceLabel}>
                            {Math.round((item.signal_confidence || 0.5) * 100)}% confidence
                        </Text>
                    </View>
                    <SignalBadge signal={item.signal} />
                </View>
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
    topRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: Spacing.sm,
    },
    left: { flexDirection: 'row', alignItems: 'center', flex: 1, marginRight: Spacing.sm },
    symbolChip: {
        width: 46, height: 46, borderRadius: Radius.md,
        backgroundColor: C.brand.glow,
        borderWidth: 1, borderColor: C.border.brand,
        alignItems: 'center', justifyContent: 'center', flexShrink: 0,
    },
    symbolText: {
        color: C.brand.purple,
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.black,
        letterSpacing: 0.5,
    },
    nameBlock: { marginLeft: Spacing.sm, flexShrink: 1 },
    symbolLabel: {
        color: C.text.primary,
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.bold,
        marginBottom: 3,
    },
    right: { alignItems: 'flex-end', flexShrink: 0 },
    price: { color: C.text.primary, fontSize: Typography.sizes.lg, fontWeight: Typography.weights.bold },
    change: { fontSize: Typography.sizes.sm, fontWeight: Typography.weights.semibold, marginTop: 2 },
    bottomRow: {
        flexDirection: 'row', alignItems: 'center',
        justifyContent: 'space-between', marginTop: 4, gap: Spacing.sm,
    },
    confidenceWrap: { flex: 1 },
    confidenceBarBg: {
        height: 4, backgroundColor: C.bg.elevated,
        borderRadius: Radius.full, overflow: 'hidden', marginBottom: 4,
    },
    confidenceBarFill: { height: 4, borderRadius: Radius.full },
    confidenceLabel: { color: C.text.muted, fontSize: Typography.sizes.xs },
});
