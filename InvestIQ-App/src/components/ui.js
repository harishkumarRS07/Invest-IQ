/**
 * Reusable UI Components – InvestIQ
 *
 * Exported:
 *  - Card           – rounded surface with shadow
 *  - SignalBadge    – BUY/SELL/HOLD colored pill
 *  - RiskBadge      – Low/Medium/High pill
 *  - GradientButton – brand CTA with purple gradient
 *  - LoadingSpinner – animated ring spinner with label
 *  - LoadingSkeleton – alias for LoadingSpinner (legacy)
 *  - EmptyState     – placeholder for empty lists
 *  - ErrorBanner    – inline error message
 *  - IndicatorRow   – key/value row for technical indicators
 */
import React, { useEffect, useRef, useMemo } from 'react';
import {
    View, Text, TouchableOpacity, Animated, StyleSheet,
    Dimensions, ActivityIndicator, Easing,
} from 'react-native';
import { Colors, Spacing, Radius, Typography, Shadow } from '../constants/theme';
import { useColors } from '../context/ThemeContext';

const W = Dimensions.get('window').width;

// ─── Card ─────────────────────────────────────────────────────────────────────
export function Card({ children, style }) {
    const C = useColors();
    return (
        <View style={[styles.card, Shadow.card, { backgroundColor: C.bg.card, borderColor: C.border.default }, style]}>
            {children}
        </View>
    );
}

// ─── SignalBadge ───────────────────────────────────────────────────────────────
const SIGNAL_COLORS = {
    BUY: { bg: Colors.signal.buyBg, text: Colors.signal.buy },
    SELL: { bg: Colors.signal.sellBg, text: Colors.signal.sell },
    HOLD: { bg: Colors.signal.holdBg, text: Colors.signal.hold },
};

export function SignalBadge({ signal }) {
    const key = (signal || 'HOLD').toUpperCase();
    const colors = SIGNAL_COLORS[key] || SIGNAL_COLORS.HOLD;
    return (
        <View style={[styles.badge, { backgroundColor: colors.bg }]}>
            <Text style={[styles.badgeText, { color: colors.text }]}>{key}</Text>
        </View>
    );
}

// ─── RiskBadge ────────────────────────────────────────────────────────────────
const RISK_COLORS = {
    Low: Colors.risk.low,
    Medium: Colors.risk.medium,
    High: Colors.risk.high,
};

export function RiskBadge({ level }) {
    const color = RISK_COLORS[level] || Colors.risk.medium;
    return (
        <View style={[styles.riskDot, { backgroundColor: color + '22' }]}>
            <View style={[styles.riskDotInner, { backgroundColor: color }]} />
            <Text style={[styles.riskText, { color }]}>{level}</Text>
        </View>
    );
}

// ─── GradientButton ─────────────────────────────────────────────────────────
export function GradientButton({ label, onPress, loading = false, style }) {
    return (
        <TouchableOpacity
            activeOpacity={0.8}
            onPress={onPress}
            disabled={loading}
            style={[styles.gradBtn, style]}
        >
            {loading
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.gradBtnText}>{label}</Text>
            }
        </TouchableOpacity>
    );
}

// ─── SecondaryButton ─────────────────────────────────────────────────────────
export function SecondaryButton({ label, onPress, style }) {
    return (
        <TouchableOpacity
            activeOpacity={0.7}
            onPress={onPress}
            style={[styles.secBtn, style]}
        >
            <Text style={styles.secBtnText}>{label}</Text>
        </TouchableOpacity>
    );
}

// ─── LoadingSpinner ───────────────────────────────────────────────────────────
export function LoadingSpinner({ label = 'Loading signals…', size = 80 }) {
    const rotate = useRef(new Animated.Value(0)).current;
    const rotateSlow = useRef(new Animated.Value(0)).current;
    const pulse = useRef(new Animated.Value(0.6)).current;

    useEffect(() => {
        // Fast outer ring
        Animated.loop(
            Animated.timing(rotate, {
                toValue: 1, duration: 900,
                easing: Easing.linear, useNativeDriver: true,
            })
        ).start();
        // Slow inner ring (counter-clockwise)
        Animated.loop(
            Animated.timing(rotateSlow, {
                toValue: -1, duration: 2200,
                easing: Easing.linear, useNativeDriver: true,
            })
        ).start();
        // Pulse glow
        Animated.loop(
            Animated.sequence([
                Animated.timing(pulse, { toValue: 1, duration: 700, useNativeDriver: true }),
                Animated.timing(pulse, { toValue: 0.6, duration: 700, useNativeDriver: true }),
            ])
        ).start();
    }, []);

    const spinFast = rotate.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });
    const spinSlow = rotateSlow.interpolate({ inputRange: [-1, 0], outputRange: ['-360deg', '0deg'] });

    const r = size / 2;

    return (
        <View style={styles.spinnerWrap}>
            {/* Outer fast ring */}
            <Animated.View style={[
                styles.spinnerRing,
                { width: size, height: size, borderRadius: r, transform: [{ rotate: spinFast }] },
                { borderColor: Colors.brand.purple, borderTopColor: 'transparent', borderLeftColor: 'transparent' },
            ]} />
            {/* Inner slow ring */}
            <Animated.View style={[
                styles.spinnerRing,
                {
                    width: size * 0.65, height: size * 0.65, borderRadius: r * 0.65,
                    transform: [{ rotate: spinSlow }], position: 'absolute'
                },
                { borderColor: Colors.brand.purple + '80', borderTopColor: 'transparent', borderRightColor: 'transparent' },
            ]} />
            {/* Pulsing center dot */}
            <Animated.View style={[
                styles.spinnerDot,
                { opacity: pulse },
                { width: size * 0.22, height: size * 0.22, borderRadius: size * 0.11 },
            ]} />
            {label ? <Text style={styles.spinnerLabel}>{label}</Text> : null}
        </View>
    );
}

// Legacy alias – keeps existing usages working
export function LoadingSkeleton({ label } = {}) {
    return <LoadingSpinner label={label} />;
}

// ─── EmptyState ───────────────────────────────────────────────────────────────
export function EmptyState({ emoji = '📊', title, subtitle }) {
    const C = useColors();
    return (
        <View style={styles.empty}>
            <Text style={styles.emptyEmoji}>{emoji}</Text>
            <Text style={[styles.emptyTitle, { color: C.text.primary }]}>{title}</Text>
            {subtitle && <Text style={[styles.emptySubtitle, { color: C.text.secondary }]}>{subtitle}</Text>}
        </View>
    );
}

// ─── ErrorBanner ─────────────────────────────────────────────────────────────
export function ErrorBanner({ message, onRetry }) {
    return (
        <View style={styles.errorBanner}>
            <Text style={styles.errorText}>⚠️ {message}</Text>
            {onRetry && (
                <TouchableOpacity onPress={onRetry}>
                    <Text style={styles.retryText}>Retry</Text>
                </TouchableOpacity>
            )}
        </View>
    );
}

// ─── IndicatorRow ─────────────────────────────────────────────────────────────
export function IndicatorRow({ label, value, unit = '', highlight = false }) {
    const C = useColors();
    const display = value !== null && value !== undefined
        ? typeof value === 'number' ? value.toFixed(2) : value
        : '—';
    return (
        <View style={[styles.indicatorRow, { borderBottomColor: C.border.subtle }]}>
            <Text style={[styles.indicatorLabel, { color: C.text.secondary }]}>{label}</Text>
            <Text style={[styles.indicatorValue, { color: highlight ? Colors.brand.purple : C.text.primary }]}>
                {display}{unit}
            </Text>
        </View>
    );
}

// ─── Section Header ──────────────────────────────────────────────────────────
export function SectionHeader({ title, action, onAction }) {
    const C = useColors();
    return (
        <View style={styles.sectionHeader}>
            <Text style={[styles.sectionTitle, { color: C.text.primary }]}>{title}</Text>
            {action && (
                <TouchableOpacity onPress={onAction}>
                    <Text style={styles.sectionAction}>{action}</Text>
                </TouchableOpacity>
            )}
        </View>
    );
}

// ─── Styles ─────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
    card: {
        borderRadius: Radius.lg,
        padding: Spacing.md,
        borderWidth: 1,
        marginBottom: Spacing.sm,
    },
    badge: {
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: Radius.full,
    },
    badgeText: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
        letterSpacing: 0.8,
    },
    riskDot: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 8,
        paddingVertical: 3,
        borderRadius: Radius.full,
        gap: 4,
    },
    riskDotInner: {
        width: 6,
        height: 6,
        borderRadius: 3,
    },
    riskText: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.semibold,
    },
    gradBtn: {
        backgroundColor: Colors.brand.purple,
        borderRadius: Radius.lg,
        height: 52,
        alignItems: 'center',
        justifyContent: 'center',
        ...Shadow.glow,
    },
    gradBtnText: {
        color: '#fff',
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.bold,
        letterSpacing: 0.5,
    },
    secBtn: {
        borderWidth: 1,
        borderColor: Colors.brand.purple,
        borderRadius: Radius.lg,
        height: 52,
        alignItems: 'center',
        justifyContent: 'center',
    },
    secBtnText: {
        color: Colors.brand.purple,
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.semibold,
    },
    // spinner
    spinnerWrap: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: Spacing.xxl,
        gap: Spacing.lg,
    },
    spinnerRing: {
        borderWidth: 3,
        borderStyle: 'solid',
    },
    spinnerDot: {
        backgroundColor: Colors.brand.purple,
        position: 'absolute',
        shadowColor: Colors.brand.purple,
        shadowOpacity: 0.9,
        shadowRadius: 8,
        elevation: 4,
    },
    spinnerLabel: {
        color: Colors.text.secondary,
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.medium,
        letterSpacing: 0.3,
        marginTop: Spacing.sm,
    },
    empty: {
        alignItems: 'center',
        paddingVertical: Spacing.xxl,
    },
    emptyEmoji: {
        fontSize: 48,
        marginBottom: Spacing.md,
    },
    emptyTitle: {
        fontSize: Typography.sizes.lg,
        fontWeight: Typography.weights.bold,
        marginBottom: Spacing.xs,
    },
    emptySubtitle: {
        fontSize: Typography.sizes.md,
        textAlign: 'center',
    },
    errorBanner: {
        backgroundColor: Colors.signal.sellBg,
        borderRadius: Radius.md,
        padding: Spacing.md,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.sm,
    },
    errorText: {
        color: Colors.signal.sell,
        fontSize: Typography.sizes.sm,
        flex: 1,
    },
    retryText: {
        color: Colors.signal.sell,
        fontWeight: Typography.weights.bold,
        marginLeft: Spacing.sm,
    },
    indicatorRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingVertical: 8,
        borderBottomWidth: 1,
    },
    indicatorLabel: {
        fontSize: Typography.sizes.sm,
    },
    indicatorValue: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
    },
    sectionHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.sm,
    },
    sectionTitle: {
        fontSize: Typography.sizes.lg,
        fontWeight: Typography.weights.bold,
    },
    sectionAction: {
        color: Colors.brand.purple,
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
    },
});
