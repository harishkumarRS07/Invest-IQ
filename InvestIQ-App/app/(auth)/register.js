/**
 * Register Screen - InvestIQ
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


const FIELDS = [
    { key: 'name', label: 'Full Name', placeholder: 'John Doe', type: 'default', cap: 'words' },
    { key: 'email', label: 'Email', placeholder: 'you@example.com', type: 'email-address', cap: 'none' },
    { key: 'password', label: 'Password', placeholder: '••••••••', secure: true, cap: 'none' },
    { key: 'confirm', label: 'Confirm Password', placeholder: '••••••••', secure: true, cap: 'none' },
];

export default function RegisterScreen() {
    const { register } = useAuth();
    const router = useRouter();
    const [values, setValues] = useState({ name: '', email: '', password: '', confirm: '' });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [visible, setVisible] = useState({ password: false, confirm: false });
    const [focused, setFocused] = useState(null);
    const insets = useSafeAreaInsets();
    const C = useColors();
    const styles = useMemo(() => makeStyles(C), [C]);

    // Animations
    const entryAnim = useRef(new Animated.Value(0)).current;
    const blob1Anim = useRef(new Animated.Value(0)).current;
    const blob2Anim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        Animated.timing(entryAnim, {
            toValue: 1,
            duration: 900,
            easing: Easing.out(Easing.cubic),
            useNativeDriver: true,
        }).start();

        const pulsate = (anim, duration) => {
            Animated.loop(
                Animated.sequence([
                    Animated.timing(anim, { toValue: 1, duration, useNativeDriver: true }),
                    Animated.timing(anim, { toValue: 0, duration, useNativeDriver: true }),
                ])
            ).start();
        };
        pulsate(blob1Anim, 5000);
        pulsate(blob2Anim, 6500);
    }, []);

    const set = (key) => (val) => setValues((v) => ({ ...v, [key]: val }));
    const toggleVisible = (key) => setVisible((v) => ({ ...v, [key]: !v[key] }));

    const handleRegister = async () => {
        const { name, email, password, confirm } = values;
        if (!name || !email || !password) { setError('All fields are required'); return; }
        if (password !== confirm) { setError('Passwords do not match'); return; }
        if (password.length < 6) { setError('Password must be at least 6 characters'); return; }
        setLoading(true);
        setError(null);
        try {
            await register(email.trim().toLowerCase(), password, name.trim());
            router.replace('/(tabs)/dashboard');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const getStyles = (index) => {
        const start = 0.08 * index;
        const opacity = entryAnim.interpolate({
            inputRange: [start, Math.min(start + 0.3, 1)],
            outputRange: [0, 1],
            extrapolate: 'clamp',
        });
        const translateY = entryAnim.interpolate({
            inputRange: [start, Math.min(start + 0.3, 1)],
            outputRange: [15, 0],
            extrapolate: 'clamp',
        });
        return { opacity, transform: [{ translateY }] };
    };

    const blob1Style = {
        transform: [
            { scale: blob1Anim.interpolate({ inputRange: [0, 1], outputRange: [1.1, 1.3] }) },
            { translateX: blob1Anim.interpolate({ inputRange: [0, 1], outputRange: [-20, 20] }) },
        ],
        opacity: blob1Anim.interpolate({ inputRange: [0, 1], outputRange: [0.04, 0.07] }),
    };
    const blob2Style = {
        transform: [
            { scale: blob2Anim.interpolate({ inputRange: [0, 1], outputRange: [1.2, 1] }) },
            { translateY: blob2Anim.interpolate({ inputRange: [0, 1], outputRange: [0, 40] }) },
        ],
        opacity: blob2Anim.interpolate({ inputRange: [0, 1], outputRange: [0.03, 0.06] }),
    };

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            enabled={Platform.OS === 'ios'}
            keyboardVerticalOffset={Platform.OS === 'ios' ? insets.top : 0}
            style={styles.root}
        >
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
                <Animated.View style={[styles.header, getStyles(0)]}>
                    <Text style={styles.title}>Create Account</Text>
                    <Text style={styles.subtitle}>Join InvestIQ and trade smarter with AI</Text>
                </Animated.View>

                <View style={styles.form}>
                    {error && <ErrorBanner message={error} />}

                    {FIELDS.map((field, idx) => {
                        const isSecure = field.secure && !visible[field.key];
                        return (
                            <Animated.View key={field.key} style={getStyles(idx + 1.5)}>
                                <Text style={[styles.label, idx === 0 && { marginTop: 0 }]}>{field.label}</Text>
                                {field.secure ? (
                                    <View style={[styles.passwordRow, focused === field.key && styles.inputFocused]}>
                                        <TextInput
                                            style={styles.passwordInput}
                                            onFocus={() => setFocused(field.key)}
                                            onBlur={() => setFocused(null)}
                                            placeholder={field.placeholder}
                                            placeholderTextColor={C.text.muted}
                                            value={values[field.key]}
                                            onChangeText={set(field.key)}
                                            secureTextEntry={isSecure}
                                            autoCapitalize="none"
                                            returnKeyType="next"
                                        />
                                        <TouchableOpacity
                                            onPress={() => toggleVisible(field.key)}
                                            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                                        >
                                            <Text style={styles.eyeIcon}>{visible[field.key] ? '🙈' : '👁️'}</Text>
                                        </TouchableOpacity>
                                    </View>
                                ) : (
                                    <TextInput
                                        style={[styles.input, focused === field.key && styles.inputFocused]}
                                        onFocus={() => setFocused(field.key)}
                                        onBlur={() => setFocused(null)}
                                        placeholder={field.placeholder}
                                        placeholderTextColor={C.text.muted}
                                        value={values[field.key]}
                                        onChangeText={set(field.key)}
                                        keyboardType={field.type || 'default'}
                                        autoCapitalize={field.cap || 'none'}
                                        returnKeyType="next"
                                    />
                                )}
                            </Animated.View>
                        );
                    })}

                    <Animated.View style={getStyles(FIELDS.length + 2)}>
                        <GradientButton
                            label="Create Account"
                            onPress={handleRegister}
                            loading={loading}
                            style={{ marginTop: Spacing.xl }}
                        />
                        <SecondaryButton
                            label="Already have an account? Sign In"
                            onPress={() => router.back()}
                            style={{ marginTop: Spacing.md }}
                        />
                    </Animated.View>
                </View>
            </ScrollView>
        </KeyboardAvoidingView>
    );
}

const makeStyles = (C) => StyleSheet.create({
    root: { flex: 1, backgroundColor: C.bg.primary },
    scrollView: { flex: 1, backgroundColor: C.bg.primary },
    blob1: {
        position: 'absolute', top: -50, left: -100,
        width: 380, height: 380, borderRadius: 190,
        backgroundColor: C.brand.purple,
    },
    blob2: {
        position: 'absolute', bottom: 100, right: -120,
        width: 420, height: 420, borderRadius: 210,
        backgroundColor: C.brand.blue,
    },
    scroll: { flexGrow: 1, paddingHorizontal: Spacing.lg },
    header: { marginBottom: Spacing.xxl, alignItems: 'center' },
    title: {
        color: C.text.primary,
        fontSize: Typography.sizes.xxxl + 4,
        fontWeight: Typography.weights.black,
        letterSpacing: 2, textAlign: 'center',
    },
    subtitle: {
        color: C.text.secondary,
        fontSize: Typography.sizes.sm,
        marginTop: 6, letterSpacing: 0.5, textAlign: 'center',
    },
    form: {
        backgroundColor: C.bg.card,
        borderRadius: Radius.xl * 1.5,
        padding: Spacing.xl,
        borderWidth: 1, borderColor: C.border.default,
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
        height: 54, paddingHorizontal: Spacing.md,
        color: C.text.primary,
        fontSize: Typography.sizes.md,
        borderWidth: 1, borderColor: C.border.default,
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
    eyeIcon: { fontSize: 16, paddingLeft: 8 },
});
