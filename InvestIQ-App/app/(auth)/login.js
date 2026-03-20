/**
 * Login Screen - InvestIQ
 * Professional fintech login with animations.
 */
import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
    View, Text, TextInput, StyleSheet, ScrollView,
    Animated, KeyboardAvoidingView, Platform, TouchableOpacity, Easing,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useAuth } from '../../src/context/AuthContext';
import { GradientButton, SecondaryButton, ErrorBanner } from '../../src/components/ui';
import { Spacing, Radius, Typography, Shadow } from '../../src/constants/theme';
import { useColors } from '../../src/context/ThemeContext';

export default function LoginScreen() {
    const { login } = useAuth();
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [pwVisible, setPwVisible] = useState(false);
    const [focused, setFocused] = useState(null);
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    // Animations
    const entryAnim = useRef(new Animated.Value(0)).current;
    const blob1Anim = useRef(new Animated.Value(0)).current;
    const blob2Anim = useRef(new Animated.Value(0)).current;
    const insets = useSafeAreaInsets();

    useEffect(() => {
        // Entry animation sequence
        Animated.timing(entryAnim, {
            toValue: 1,
            duration: 800,
            easing: Easing.out(Easing.cubic),
            useNativeDriver: true,
        }).start();

        // Pulsating background blobs
        const pulsate = (anim, duration) => {
            Animated.loop(
                Animated.sequence([
                    Animated.timing(anim, { toValue: 1, duration, useNativeDriver: true }),
                    Animated.timing(anim, { toValue: 0, duration, useNativeDriver: true }),
                ])
            ).start();
        };
        pulsate(blob1Anim, 4000);
        pulsate(blob2Anim, 5500);
    }, []);

    const handleLogin = async () => {
        if (!email || !password) { setError('Please fill in all fields'); return; }
        setLoading(true);
        setError(null);
        try {
            await login(email.trim().toLowerCase(), password);
            router.replace('/(tabs)/dashboard');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Staggered interpolations
    const getStyles = (index) => {
        const start = 0.1 * index;
        const opacity = entryAnim.interpolate({
            inputRange: [start, Math.min(start + 0.4, 1)],
            outputRange: [0, 1],
            extrapolate: 'clamp',
        });
        const translateY = entryAnim.interpolate({
            inputRange: [start, Math.min(start + 0.4, 1)],
            outputRange: [20, 0],
            extrapolate: 'clamp',
        });
        return { opacity, transform: [{ translateY }] };
    };

    const logoStyle = getStyles(0);
    const emailStyle = getStyles(1);
    const passStyle = getStyles(2);
    const btnStyle = getStyles(3);
    const footStyle = getStyles(4);

    const blob1Style = {
        transform: [
            { scale: blob1Anim.interpolate({ inputRange: [0, 1], outputRange: [1, 1.2] }) },
            { translateX: blob1Anim.interpolate({ inputRange: [0, 1], outputRange: [0, 20] }) },
        ],
        opacity: blob1Anim.interpolate({ inputRange: [0, 1], outputRange: [0.03, 0.08] }),
    };
    const blob2Style = {
        transform: [
            { scale: blob2Anim.interpolate({ inputRange: [0, 1], outputRange: [1.3, 1] }) },
            { translateY: blob2Anim.interpolate({ inputRange: [0, 1], outputRange: [0, -30] }) },
        ],
        opacity: blob2Anim.interpolate({ inputRange: [0, 1], outputRange: [0.04, 0.09] }),
    };

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            enabled={Platform.OS === 'ios'}
            keyboardVerticalOffset={Platform.OS === 'ios' ? insets.top : 0}
            style={styles.root}
        >
            {/* Background Decorations */}
            <Animated.View style={[styles.blob1, blob1Style]} />
            <Animated.View style={[styles.blob2, blob2Style]} />

            <ScrollView
                style={styles.scrollView}
                contentContainerStyle={[
                    styles.scroll,
                    { paddingTop: insets.top + Spacing.xl, paddingBottom: insets.bottom + Spacing.xl }
                ]}
                keyboardShouldPersistTaps="handled"
                showsVerticalScrollIndicator={false}
            >
                {/* Logo / Branding */}
                <Animated.View style={[styles.logoBlock, logoStyle]}>
                    <View style={styles.logoCircle}>
                        <Text style={styles.logoEmoji}>📈</Text>
                    </View>
                    <Text style={styles.appName}>InvestIQ</Text>
                    <Text style={styles.tagline}>AI-Powered Trading Intelligence</Text>
                </Animated.View>

                {/* Form */}
                <Animated.View style={styles.form}>
                    {error && <ErrorBanner message={error} />}

                    <Animated.View style={emailStyle}>
                        <Text style={[styles.label, { marginTop: 0 }]}>Email</Text>
                        <TextInput
                            style={[styles.input, focused === 'email' && styles.inputFocused]}
                            onFocus={() => setFocused('email')}
                            onBlur={() => setFocused(null)}
                            placeholder="you@example.com"
                            placeholderTextColor={C.text.muted}
                            value={email}
                            onChangeText={setEmail}
                            autoCapitalize="none"
                            keyboardType="email-address"
                            returnKeyType="next"
                        />
                    </Animated.View>

                    <Animated.View style={passStyle}>
                        <Text style={styles.label}>Password</Text>
                        <View style={[styles.passwordRow, focused === 'pass' && styles.inputFocused]}>
                            <TextInput
                                style={styles.passwordInput}
                                onFocus={() => setFocused('pass')}
                                onBlur={() => setFocused(null)}
                                placeholder="••••••••"
                                placeholderTextColor={C.text.muted}
                                value={password}
                                onChangeText={setPassword}
                                secureTextEntry={!pwVisible}
                                returnKeyType="done"
                                onSubmitEditing={handleLogin}
                            />
                            <TouchableOpacity
                                style={styles.eyeBtn}
                                onPress={() => setPwVisible((v) => !v)}
                                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                            >
                                <Text style={styles.eyeIcon}>{pwVisible ? '🙈' : '👁️'}</Text>
                            </TouchableOpacity>
                        </View>
                    </Animated.View>

                    <Animated.View style={btnStyle}>
                        <GradientButton
                            label="Sign In"
                            onPress={handleLogin}
                            loading={loading}
                            style={{ marginTop: Spacing.xl }}
                        />
                        <SecondaryButton
                            label="Create Account"
                            onPress={() => router.push('/(auth)/register')}
                            style={{ marginTop: Spacing.md }}
                        />
                    </Animated.View>
                </Animated.View>

                <Animated.Text style={[styles.footer, footStyle]}>
                    Professional AI trading signals for modern investors.{"\n"}
                    Demo: Register for free to start.
                </Animated.Text>
            </ScrollView>
        </KeyboardAvoidingView>
    );
}

const makeStyles = (C) => StyleSheet.create({
    root: { flex: 1, backgroundColor: C.bg.primary },
    scrollView: { flex: 1, backgroundColor: C.bg.primary },
    blob1: {
        position: 'absolute', top: -100, right: -100,
        width: 400, height: 400, borderRadius: 200,
        backgroundColor: C.brand.purple,
    },
    blob2: {
        position: 'absolute', bottom: -50, left: -100,
        width: 350, height: 350, borderRadius: 175,
        backgroundColor: C.brand.blue,
    },
    scroll: { flexGrow: 1, paddingHorizontal: Spacing.lg, justifyContent: 'center' },
    logoBlock: { alignItems: 'center', marginBottom: Spacing.xxl },
    logoCircle: {
        width: 84, height: 84, borderRadius: 42,
        backgroundColor: C.brand.purple,
        alignItems: 'center', justifyContent: 'center',
        marginBottom: Spacing.md, ...Shadow.glow,
    },
    logoEmoji: { fontSize: 38 },
    appName: {
        color: C.text.primary,
        fontSize: Typography.sizes.xxxl + 4,
        fontWeight: Typography.weights.black,
        letterSpacing: 2,
    },
    tagline: { color: C.text.secondary, fontSize: Typography.sizes.sm, marginTop: 6, letterSpacing: 0.5 },
    form: {
        backgroundColor: C.bg.card,
        borderRadius: Radius.xl * 1.5,
        padding: Spacing.xl,
        borderWidth: 1,
        borderColor: C.border.default,
        ...Shadow.card,
    },
    label: {
        color: C.text.secondary,
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
        marginBottom: 8, marginTop: Spacing.lg,
    },
    input: {
        backgroundColor: C.bg.input,
        borderRadius: Radius.md,
        height: 54,
        paddingHorizontal: Spacing.md,
        color: C.text.primary,
        fontSize: Typography.sizes.md,
        borderWidth: 1,
        borderColor: C.border.default,
    },
    inputFocused: { borderColor: C.brand.purple, borderWidth: 1.5 },
    passwordRow: {
        flexDirection: 'row', alignItems: 'center',
        backgroundColor: C.bg.input,
        borderRadius: Radius.md,
        borderWidth: 1, borderColor: C.border.default,
        height: 54, paddingHorizontal: Spacing.md,
    },
    passwordInput: { flex: 1, color: C.text.primary, fontSize: Typography.sizes.md },
    eyeBtn: { paddingLeft: 8 },
    eyeIcon: { fontSize: 16 },
    footer: {
        color: C.text.muted,
        fontSize: Typography.sizes.xs,
        textAlign: 'center',
        marginTop: Spacing.xl, marginBottom: Spacing.lg,
        lineHeight: 18, paddingHorizontal: Spacing.lg,
    },
});
