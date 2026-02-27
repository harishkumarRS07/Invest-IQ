/**
 * FloatingIQMenu.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Three states:
 *  HIDDEN   → only pull-tab arrow at right edge
 *  REVEALED → IQ button slides in (arrow flips)
 *  EXPANDED → nav items appear above IQ button (tap IQ or background to close)
 *
 * Reliable architecture:
 *  • Items use flex layout (not transforms) → correct Android hitboxes
 *  • Overlay rendered before items → items always win touch disputes
 *  • navigation.navigate(tabName) from tabBar props, router fallback
 *  • Fast-switch guard via useRef (never causes re-render)
 * ─────────────────────────────────────────────────────────────────────────────
 */

import React, { useRef, useState, useCallback } from 'react';
import {
    View, Text, TouchableOpacity, Pressable, Animated,
    StyleSheet, Dimensions,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

const { width: W } = Dimensions.get('window');

// ─── Design tokens ────────────────────────────────────────────────────────────
const C = {
    purple: '#7B61FF',
    purpleLt: '#9B84FF',
    purpleGlow: 'rgba(123,97,255,0.30)',
    purpleRing: 'rgba(123,97,255,0.14)',
    btn: '#1A2035',
    btnBorder: 'rgba(123,97,255,0.40)',
    muted: '#8B9DC3',
    overlay: 'rgba(4,8,20,0.70)',
};

const IQ_SIZE = 68;
const IQ_RADIUS = IQ_SIZE / 2;
const ITEM_SIZE = 54;
const TAB_W = 22;   // pull-tab width
const TAB_H = 52;   // pull-tab height

// When hidden, slide the menu this far to the right (off-screen)
const HIDDEN_TX = IQ_SIZE + 28 + 16;   // IQ width + right margin + slack

// ─── Tabs ─────────────────────────────────────────────────────────────────────
const TABS = [
    { name: 'dashboard', emoji: '📊', label: 'Market' },
    { name: 'portfolio', emoji: '💼', label: 'Portfolio' },
    { name: 'settings', emoji: '⚙️', label: 'Settings' },
];

// ─── Component ────────────────────────────────────────────────────────────────
export default function FloatingIQMenu({ navigation, state }) {
    const router = useRouter();
    const insets = useSafeAreaInsets();

    // Two independent booleans keep state logic simple
    const [visible, setVisible] = useState(false);  // IQ button on/off screen
    const [expanded, setExpanded] = useState(false);  // nav items visible

    const busy = useRef(false);

    // ── Animated values ──────────────────────────────────────────────────────
    const slideX = useRef(new Animated.Value(HIDDEN_TX)).current;  // menu x
    const arrowRot = useRef(new Animated.Value(0)).current;           // pull-tab arrow
    const overlayOp = useRef(new Animated.Value(0)).current;
    const itemsOp = useRef(new Animated.Value(0)).current;
    const itemsScale = useRef(new Animated.Value(0.75)).current;
    const iqRot = useRef(new Animated.Value(0)).current;
    const pulseScale = useRef(new Animated.Value(1)).current;
    const pulseOp = useRef(new Animated.Value(0.6)).current;
    const pulseLoop = useRef(null);

    // ── Pulse glow while IQ idle ──────────────────────────────────────────────
    const startPulse = useCallback(() => {
        pulseLoop.current?.stop();
        pulseLoop.current = Animated.loop(
            Animated.sequence([
                Animated.parallel([
                    Animated.timing(pulseScale, { toValue: 1.25, duration: 1000, useNativeDriver: true }),
                    Animated.timing(pulseOp, { toValue: 0, duration: 1000, useNativeDriver: true }),
                ]),
                Animated.parallel([
                    Animated.timing(pulseScale, { toValue: 1, duration: 0, useNativeDriver: true }),
                    Animated.timing(pulseOp, { toValue: 0.6, duration: 0, useNativeDriver: true }),
                ]),
            ])
        );
        pulseLoop.current.start();
    }, [pulseScale, pulseOp]);

    const stopPulse = useCallback(() => {
        pulseLoop.current?.stop();
        pulseScale.setValue(1);
        pulseOp.setValue(0);
    }, [pulseScale, pulseOp]);

    // ── Show / hide IQ button (slide in/out) ─────────────────────────────────
    const showMenu = useCallback(() => {
        setVisible(true);
        Animated.parallel([
            Animated.spring(slideX, { toValue: 0, damping: 16, stiffness: 140, useNativeDriver: true }),
            Animated.spring(arrowRot, { toValue: 1, damping: 14, stiffness: 180, useNativeDriver: true }),
        ]).start(() => startPulse());
    }, [slideX, arrowRot, startPulse]);

    const hideMenu = useCallback(() => {
        // Collapse items first if expanded
        if (expanded) {
            Animated.parallel([
                Animated.timing(overlayOp, { toValue: 0, duration: 150, useNativeDriver: true }),
                Animated.timing(itemsOp, { toValue: 0, duration: 120, useNativeDriver: true }),
                Animated.timing(itemsScale, { toValue: 0.75, duration: 120, useNativeDriver: true }),
                Animated.timing(iqRot, { toValue: 0, duration: 150, useNativeDriver: true }),
            ]).start();
            setExpanded(false);
        }
        stopPulse();
        Animated.parallel([
            Animated.spring(slideX, { toValue: HIDDEN_TX, damping: 16, stiffness: 140, useNativeDriver: true }),
            Animated.spring(arrowRot, { toValue: 0, damping: 14, stiffness: 180, useNativeDriver: true }),
        ]).start(() => setVisible(false));
    }, [slideX, arrowRot, expanded, overlayOp, itemsOp, itemsScale, iqRot, stopPulse]);

    const toggleVisible = useCallback(() => {
        visible ? hideMenu() : showMenu();
    }, [visible, showMenu, hideMenu]);

    // ── Expand / collapse nav items ───────────────────────────────────────────
    const openItems = useCallback(() => {
        stopPulse();
        setExpanded(true);
        Animated.parallel([
            Animated.timing(overlayOp, { toValue: 1, duration: 200, useNativeDriver: true }),
            Animated.timing(itemsOp, { toValue: 1, duration: 220, useNativeDriver: true }),
            Animated.spring(itemsScale, { toValue: 1, damping: 14, stiffness: 200, useNativeDriver: true }),
            Animated.spring(iqRot, { toValue: 1, damping: 14, stiffness: 180, useNativeDriver: true }),
        ]).start();
    }, [overlayOp, itemsOp, itemsScale, iqRot, stopPulse]);

    const closeItems = useCallback(() => {
        Animated.parallel([
            Animated.timing(overlayOp, { toValue: 0, duration: 180, useNativeDriver: true }),
            Animated.timing(itemsOp, { toValue: 0, duration: 150, useNativeDriver: true }),
            Animated.timing(itemsScale, { toValue: 0.75, duration: 150, useNativeDriver: true }),
            Animated.spring(iqRot, { toValue: 0, damping: 14, stiffness: 180, useNativeDriver: true }),
        ]).start(({ finished }) => { if (finished) { setExpanded(false); startPulse(); } });
    }, [overlayOp, itemsOp, itemsScale, iqRot, startPulse]);

    const toggleExpanded = useCallback(() => {
        expanded ? closeItems() : openItems();
    }, [expanded, openItems, closeItems]);

    // ── Navigate — menu stays open to allow free tab switching ───────────────
    const goTo = useCallback((tabName) => {
        if (busy.current) return;
        busy.current = true;
        if (navigation?.navigate) {
            navigation.navigate(tabName);
        } else {
            const routes = {
                dashboard: '/(tabs)/dashboard',
                portfolio: '/(tabs)/portfolio',
                settings: '/(tabs)/settings',
            };
            router.navigate(routes[tabName]);
        }
        setTimeout(() => { busy.current = false; }, 350);
    }, [navigation, router]);

    // ── Interpolations ────────────────────────────────────────────────────────
    const arrowRotI = arrowRot.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '180deg'] });
    const iqRotI = iqRot.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '45deg'] });

    const activeTab = state?.routes?.[state.index]?.name ?? 'dashboard';
    const bottom = insets.bottom + 20;

    // ─────────────────────────────────────────────────────────────────────────
    return (
        <>
            {/* Overlay — rendered FIRST (lowest touch priority) */}
            {expanded && (
                <Animated.View
                    style={[styles.overlay, { opacity: overlayOp }]}
                    pointerEvents="box-none"
                >
                    <Pressable style={StyleSheet.absoluteFill} onPress={closeItems} />
                </Animated.View>
            )}

            {/*
             * Pull-tab — fixed at right edge, always visible.
             * Tapping toggles IQ button visibility.
             */}
            <TouchableOpacity
                onPress={toggleVisible}
                activeOpacity={0.75}
                style={[styles.pullTab, { bottom: bottom + (IQ_RADIUS - TAB_H / 2) }]}
            >
                <Animated.Text style={[styles.pullArrow, { transform: [{ rotate: arrowRotI }] }]}>
                    ❮
                </Animated.Text>
            </TouchableOpacity>

            {/*
             * Menu container — slides in/out via translateX.
             * Contains items (flex layout) and IQ button.
             * Rendered AFTER overlay → higher touch priority.
             */}
            <Animated.View
                style={[styles.menuContainer, { bottom, transform: [{ translateX: slideX }] }]}
                pointerEvents="box-none"
            >
                {/* Nav items row — shown when expanded */}
                {(visible && expanded) && (
                    <Animated.View
                        style={[styles.itemsRow, { opacity: itemsOp, transform: [{ scale: itemsScale }] }]}
                        pointerEvents="box-none"
                    >
                        {TABS.map((tab) => {
                            const isActive = activeTab === tab.name;
                            return (
                                <View key={tab.name} style={styles.itemWrapper}>
                                    {isActive && <View style={styles.activeRing} />}
                                    <TouchableOpacity
                                        onPress={() => goTo(tab.name)}
                                        activeOpacity={0.75}
                                        style={[styles.itemBtn, isActive && styles.itemBtnActive]}
                                    >
                                        <Text style={styles.itemEmoji}>{tab.emoji}</Text>
                                    </TouchableOpacity>
                                    <Text style={[styles.itemLabel, isActive && { color: C.purple }]}>
                                        {tab.label}
                                    </Text>
                                </View>
                            );
                        })}
                    </Animated.View>
                )}

                {/* IQ button */}
                <View style={styles.iqWrapper}>
                    <Animated.View
                        style={[styles.pulseRing, { transform: [{ scale: pulseScale }], opacity: pulseOp }]}
                        pointerEvents="none"
                    />
                    <View style={styles.glowHalo} pointerEvents="none" />
                    <TouchableOpacity
                        onPress={toggleExpanded}
                        activeOpacity={0.85}
                        style={styles.iqBtn}
                    >
                        <View style={styles.iqShimmer} />
                        <Animated.Text style={[styles.iqLabel, { transform: [{ rotate: iqRotI }] }]}>
                            IQ
                        </Animated.Text>
                    </TouchableOpacity>
                </View>
            </Animated.View>
        </>
    );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({

    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: C.overlay,
        zIndex: 40,
    },

    // Pull-tab — narrow pill at absolute right edge
    pullTab: {
        position: 'absolute',
        right: 0,
        width: TAB_W,
        height: TAB_H,
        backgroundColor: C.purple,
        borderTopLeftRadius: 12,
        borderBottomLeftRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 60,
        shadowColor: C.purple,
        shadowOffset: { width: -3, height: 0 },
        shadowOpacity: 0.7,
        shadowRadius: 8,
        elevation: 14,
    },
    pullArrow: {
        fontSize: 13,
        color: '#fff',
        fontWeight: '800',
    },

    // Sliding menu container
    menuContainer: {
        position: 'absolute',
        right: 24,
        alignItems: 'center',
        zIndex: 50,
    },

    // Items arranged in a horizontal row above IQ
    itemsRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        marginBottom: 14,
        gap: 10,
    },
    itemWrapper: {
        alignItems: 'center',
    },
    activeRing: {
        position: 'absolute',
        top: -5,
        left: -5,
        width: ITEM_SIZE + 10,
        height: ITEM_SIZE + 10,
        borderRadius: (ITEM_SIZE + 10) / 2,
        borderWidth: 1.5,
        borderColor: C.purple,
        backgroundColor: C.purpleGlow,
    },
    itemBtn: {
        width: ITEM_SIZE,
        height: ITEM_SIZE,
        borderRadius: ITEM_SIZE / 2,
        backgroundColor: C.btn,
        borderWidth: 1,
        borderColor: C.btnBorder,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.5,
        shadowRadius: 10,
        elevation: 10,
    },
    itemBtnActive: {
        backgroundColor: 'rgba(123,97,255,0.18)',
        borderColor: C.purple,
    },
    itemEmoji: { fontSize: 22 },
    itemLabel: {
        marginTop: 5,
        fontSize: 10,
        fontWeight: '700',
        color: C.muted,
        letterSpacing: 0.4,
        textShadowColor: 'rgba(0,0,0,0.7)',
        textShadowOffset: { width: 0, height: 1 },
        textShadowRadius: 3,
    },

    // IQ button + glow
    iqWrapper: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    pulseRing: {
        position: 'absolute',
        width: IQ_SIZE + 24,
        height: IQ_SIZE + 24,
        borderRadius: (IQ_SIZE + 24) / 2,
        backgroundColor: C.purpleRing,
    },
    glowHalo: {
        position: 'absolute',
        width: IQ_SIZE + 14,
        height: IQ_SIZE + 14,
        borderRadius: (IQ_SIZE + 14) / 2,
        backgroundColor: C.purpleGlow,
    },
    iqBtn: {
        width: IQ_SIZE,
        height: IQ_SIZE,
        borderRadius: IQ_RADIUS,
        backgroundColor: C.purple,
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: C.purpleLt,
        shadowColor: C.purple,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.9,
        shadowRadius: 22,
        elevation: 20,
    },
    iqShimmer: {
        position: 'absolute',
        top: 0, left: 0, right: 0,
        height: '50%',
        backgroundColor: 'rgba(255,255,255,0.13)',
        borderTopLeftRadius: IQ_RADIUS,
        borderTopRightRadius: IQ_RADIUS,
    },
    iqLabel: {
        fontSize: 20,
        fontWeight: '900',
        color: '#FFF',
        letterSpacing: 2,
        textShadowColor: 'rgba(0,0,0,0.35)',
        textShadowOffset: { width: 0, height: 1 },
        textShadowRadius: 4,
    },
});
