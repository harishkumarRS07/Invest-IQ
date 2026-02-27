/**
 * InvestIQ – In-App Animated Splash Screen
 * Shown after the native Expo splash, before auth screens.
 *
 * Animation sequence:
 *  0ms  → logo circle scales up + fades in
 *  300ms → app name slides up + fades in
 *  600ms → tagline fades in
 *  900ms → pulse ring expands
 *  2400ms → whole screen fades out
 */
import React, { useEffect, useRef } from 'react';
import {
    View, Text, StyleSheet, Animated, Dimensions, Easing,
} from 'react-native';
import { Colors, Typography } from '../constants/theme';

const { width, height } = Dimensions.get('window');

export default function AnimatedSplash({ onFinish }) {
    // Core animations
    const logoScale = useRef(new Animated.Value(0)).current;
    const logoOpacity = useRef(new Animated.Value(0)).current;
    const nameY = useRef(new Animated.Value(30)).current;
    const nameOpacity = useRef(new Animated.Value(0)).current;
    const tagOpacity = useRef(new Animated.Value(0)).current;
    const ringScale = useRef(new Animated.Value(0.6)).current;
    const ringOpacity = useRef(new Animated.Value(0.7)).current;
    const screenOp = useRef(new Animated.Value(1)).current;

    useEffect(() => {
        Animated.sequence([
            // 1. Logo pops in
            Animated.parallel([
                Animated.spring(logoScale, {
                    toValue: 1,
                    friction: 5,
                    tension: 80,
                    useNativeDriver: true,
                }),
                Animated.timing(logoOpacity, {
                    toValue: 1,
                    duration: 300,
                    useNativeDriver: true,
                }),
            ]),
            // 2. App name slides up
            Animated.parallel([
                Animated.timing(nameY, {
                    toValue: 0,
                    duration: 350,
                    easing: Easing.out(Easing.cubic),
                    useNativeDriver: true,
                }),
                Animated.timing(nameOpacity, {
                    toValue: 1,
                    duration: 350,
                    useNativeDriver: true,
                }),
            ]),
            // 3. Tagline fades in
            Animated.timing(tagOpacity, {
                toValue: 1,
                duration: 300,
                useNativeDriver: true,
            }),
            // 4. Pulse ring expands
            Animated.parallel([
                Animated.timing(ringScale, {
                    toValue: 2.8,
                    duration: 800,
                    easing: Easing.out(Easing.quad),
                    useNativeDriver: true,
                }),
                Animated.timing(ringOpacity, {
                    toValue: 0,
                    duration: 800,
                    useNativeDriver: true,
                }),
            ]),
            // 5. Hold briefly
            Animated.delay(300),
            // 6. Fade entire screen out
            Animated.timing(screenOp, {
                toValue: 0,
                duration: 500,
                easing: Easing.in(Easing.quad),
                useNativeDriver: true,
            }),
        ]).start(() => {
            onFinish?.();
        });
    }, []);

    return (
        <Animated.View style={[styles.root, { opacity: screenOp }]}>
            {/* Background gradient dots (static decoration) */}
            <View style={styles.bgDot1} />
            <View style={styles.bgDot2} />

            {/* Centre content */}
            <View style={styles.centre}>
                {/* Pulse ring behind logo */}
                <Animated.View
                    style={[
                        styles.pulseRing,
                        { opacity: ringOpacity, transform: [{ scale: ringScale }] },
                    ]}
                />

                {/* Logo circle */}
                <Animated.View
                    style={[
                        styles.logoCircle,
                        { opacity: logoOpacity, transform: [{ scale: logoScale }] },
                    ]}
                >
                    <Text style={styles.logoEmoji}>📈</Text>
                </Animated.View>

                {/* App name */}
                <Animated.Text
                    style={[
                        styles.appName,
                        { opacity: nameOpacity, transform: [{ translateY: nameY }] },
                    ]}
                >
                    InvestIQ
                </Animated.Text>

                {/* Tagline */}
                <Animated.Text style={[styles.tagline, { opacity: tagOpacity }]}>
                    AI-Powered Trading Intelligence
                </Animated.Text>

                {/* Loading dots */}
                <Animated.View style={[styles.dotsRow, { opacity: tagOpacity }]}>
                    <LoadingDot delay={0} />
                    <LoadingDot delay={200} />
                    <LoadingDot delay={400} />
                </Animated.View>
            </View>

            {/* Bottom credit */}
            <Animated.Text style={[styles.credit, { opacity: tagOpacity }]}>
                Powered by AI & Machine Learning
            </Animated.Text>
        </Animated.View>
    );
}

/** A single bouncing dot for the loading indicator */
function LoadingDot({ delay }) {
    const anim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        const loop = Animated.loop(
            Animated.sequence([
                Animated.delay(delay),
                Animated.timing(anim, {
                    toValue: -8,
                    duration: 350,
                    easing: Easing.out(Easing.quad),
                    useNativeDriver: true,
                }),
                Animated.timing(anim, {
                    toValue: 0,
                    duration: 350,
                    easing: Easing.in(Easing.quad),
                    useNativeDriver: true,
                }),
                Animated.delay(600 - delay),
            ])
        );
        loop.start();
        return () => loop.stop();
    }, []);

    return (
        <Animated.View style={[styles.dot, { transform: [{ translateY: anim }] }]} />
    );
}

const styles = StyleSheet.create({
    root: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: Colors.bg.primary,
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 999,
    },
    bgDot1: {
        position: 'absolute',
        width: 320,
        height: 320,
        borderRadius: 160,
        backgroundColor: 'rgba(123,97,255,0.06)',
        top: -60,
        right: -80,
    },
    bgDot2: {
        position: 'absolute',
        width: 250,
        height: 250,
        borderRadius: 125,
        backgroundColor: 'rgba(0,212,255,0.04)',
        bottom: 40,
        left: -60,
    },
    centre: {
        alignItems: 'center',
    },
    pulseRing: {
        position: 'absolute',
        width: 100,
        height: 100,
        borderRadius: 50,
        borderWidth: 2,
        borderColor: Colors.brand.purple,
    },
    logoCircle: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: Colors.brand.purple,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: Colors.brand.purple,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.6,
        shadowRadius: 24,
        elevation: 16,
        marginBottom: 24,
    },
    logoEmoji: {
        fontSize: 44,
    },
    appName: {
        color: Colors.text.primary,
        fontSize: 42,
        fontWeight: '900',
        letterSpacing: 2,
        marginBottom: 8,
    },
    tagline: {
        color: Colors.text.secondary,
        fontSize: Typography.sizes.sm,
        letterSpacing: 0.5,
        marginBottom: 36,
    },
    dotsRow: {
        flexDirection: 'row',
        gap: 8,
        alignItems: 'flex-end',
        height: 20,
    },
    dot: {
        width: 7,
        height: 7,
        borderRadius: 3.5,
        backgroundColor: Colors.brand.purple,
    },
    credit: {
        position: 'absolute',
        bottom: 48,
        color: Colors.text.muted,
        fontSize: Typography.sizes.xs,
        letterSpacing: 0.5,
    },
});
