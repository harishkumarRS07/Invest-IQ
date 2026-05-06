/**
 * Settings Screen
 * Profile, dark mode toggle, and app info.
 */
import React, { useEffect, useState, useMemo } from 'react';
import {
    View, Text, StyleSheet, ScrollView,
    Switch, TouchableOpacity, Alert,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useAuth } from '../../src/context/AuthContext';
import { useTheme, useColors } from '../../src/context/ThemeContext';
import { Card, SectionHeader } from '../../src/components/ui';
import { Spacing, Radius, Typography } from '../../src/constants/theme';
import { healthApi } from '../../src/services/api';

export default function SettingsScreen() {
    const { user, logout } = useAuth();
    const { isDark, toggle: toggleTheme } = useTheme();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);
    const [apiStatus, setApiStatus] = useState(null);
    const insets = useSafeAreaInsets();

    useEffect(() => {
        healthApi.check().then((d) => setApiStatus(d.status)).catch(() => setApiStatus('offline'));
    }, []);

    const handleLogout = () => {
        Alert.alert('Log Out', 'Are you sure you want to sign out?', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Sign Out', style: 'destructive', onPress: logout },
        ]);
    };

    const isOnline = apiStatus === 'ok';
    const statusColor = isOnline ? C.signal.buy : C.signal.sell;
    const statusLabel = apiStatus === null ? 'Checking…' : isOnline ? 'Connected' : 'Offline';

    return (
        <View style={[styles.root, { paddingTop: insets.top }]}>
            <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
                {/* Page Header */}
                <View style={styles.pageHeader}>
                    <Text style={styles.pageTitle}>Settings</Text>
                    <Text style={styles.pageSubtitle}>⚙️ Account & Preferences</Text>
                </View>

                {/* Profile Card */}
                <Card style={styles.profileCard}>
                    <View style={styles.avatarCircle}>
                        <Text style={styles.avatarText}>
                            {(user?.name || 'U').charAt(0).toUpperCase()}
                        </Text>
                    </View>
                    <View style={styles.profileInfo}>
                        <Text style={styles.profileName} numberOfLines={1}>{user?.name || 'Trader'}</Text>
                        <Text style={styles.profileEmail} numberOfLines={1}>{user?.email || ''}</Text>
                    </View>
                </Card>

                {/* Preferences */}
                <Card>
                    <SectionHeader title="Preferences" />
                    <SettingRow
                        icon="🌙"
                        label="Dark Mode"
                        value={isDark}
                        onToggle={toggleTheme}
                        isToggle
                        C={C}
                    />
                </Card>

                {/* Connection */}
                <Card>
                    <SectionHeader title="Connection" />
                    <View style={styles.statusRow}>
                        <Text style={styles.statusLabel}>Backend API</Text>
                        <View style={styles.statusPill}>
                            <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
                            <Text style={[styles.statusText, { color: statusColor }]}>{statusLabel}</Text>
                        </View>
                    </View>
                </Card>

                {/* About */}
                <Card>
                    <SectionHeader title="About" />
                    <AboutRow label="App Version" value="1.0.0" styles={styles} />
                    <AboutRow label="AI Model" value="Hybrid (XGBoost + LSTM)" styles={styles} />
                    <AboutRow label="Data Source" value="Yahoo Finance" styles={styles} />
                    <AboutRow label="Backend" value="FastAPI + Python" styles={styles} />
                </Card>

                {/* Sign Out */}
                <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
                    <Text style={styles.logoutText}>Sign Out</Text>
                </TouchableOpacity>

                <Text style={styles.footer}>
                    InvestIQ © 2026 · AI-Powered Trading Intelligence{'\n'}
                    Fors financial advice.
                </Text>
            </ScrollView>
        </View>
    );
}

function SettingRow({ icon, label, value, onToggle, isToggle, C }) {
    return (
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12 }}>
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
                <Text style={{ fontSize: 20 }}>{icon}</Text>
                <Text style={{ color: C.text.primary, fontSize: Typography.sizes.md }}>{label}</Text>
            </View>
            {isToggle && (
                <Switch
                    value={value}
                    onValueChange={onToggle}
                    trackColor={{ false: C.bg.elevated, true: C.brand.purple }}
                    thumbColor="#fff"
                />
            )}
        </View>
    );
}

function AboutRow({ label, value, styles }) {
    return (
        <View style={styles.aboutRow}>
            <Text style={styles.aboutLabel}>{label}</Text>
            <Text style={styles.aboutValue} numberOfLines={1}>{value}</Text>
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
    pageSubtitle: { color: C.text.secondary, fontSize: Typography.sizes.md, marginTop: 4 },
    profileCard: { flexDirection: 'row', alignItems: 'center' },
    avatarCircle: {
        width: 56, height: 56, borderRadius: 28,
        backgroundColor: C.brand.purple,
        alignItems: 'center', justifyContent: 'center', flexShrink: 0,
    },
    avatarText: { color: '#fff', fontSize: Typography.sizes.xl, fontWeight: Typography.weights.black },
    profileInfo: { marginLeft: Spacing.md, flex: 1 },
    profileName: { color: C.text.primary, fontSize: Typography.sizes.lg, fontWeight: Typography.weights.bold },
    profileEmail: { color: C.text.secondary, fontSize: Typography.sizes.sm, marginTop: 2 },
    statusRow: {
        flexDirection: 'row', justifyContent: 'space-between',
        alignItems: 'center', paddingVertical: 10,
    },
    statusLabel: { color: C.text.secondary, fontSize: Typography.sizes.md },
    statusPill: { flexDirection: 'row', alignItems: 'center', gap: 6 },
    statusDot: { width: 8, height: 8, borderRadius: 4 },
    statusText: { fontSize: Typography.sizes.sm, fontWeight: Typography.weights.semibold },
    aboutRow: {
        flexDirection: 'row', justifyContent: 'space-between',
        alignItems: 'center', paddingVertical: 9,
        borderBottomWidth: 1, borderBottomColor: C.border.subtle,
    },
    aboutLabel: { color: C.text.secondary, fontSize: Typography.sizes.sm },
    aboutValue: {
        color: C.text.primary,
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
        maxWidth: '55%', textAlign: 'right',
    },
    logoutBtn: {
        backgroundColor: C.signal.sellBg,
        borderRadius: Radius.lg,
        height: 52,
        alignItems: 'center', justifyContent: 'center',
        borderWidth: 1, borderColor: C.signal.sell + '66',
        marginTop: Spacing.md,
    },
    logoutText: { color: C.signal.sell, fontSize: Typography.sizes.md, fontWeight: Typography.weights.bold },
    footer: {
        color: C.text.muted, fontSize: Typography.sizes.xs,
        textAlign: 'center', marginTop: Spacing.xl, lineHeight: 18,
    },
});
