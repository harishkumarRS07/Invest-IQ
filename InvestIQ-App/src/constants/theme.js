/**
 * Design Tokens - InvestIQ Fintech Theme
 * Dark-first with premium color palette and consistent spacing.
 */

export const Colors = {
    // ─── Backgrounds ────────────────────────────────────────────
    bg: {
        primary: '#0A0E1A',      // Deep navy - main background
        secondary: '#111827',    // Slightly lighter panel
        card: '#151E2E',         // Card surface
        elevated: '#1C2740',     // Elevated card / modal
        input: '#1A2237',        // Input field background
    },

    // ─── Brand / Accent ─────────────────────────────────────────
    brand: {
        purple: '#7B61FF',       // Primary CTA
        purpleLight: '#9B84FF',
        purpleDark: '#5B41DF',
        blue: '#4D9DE0',         // Secondary accent
        cyan: '#00D4FF',
        glow: 'rgba(123,97,255,0.25)',
    },

    // ─── Semantic ────────────────────────────────────────────────
    signal: {
        buy: '#00D07C',          // Green - BUY
        sell: '#FF5353',         // Red - SELL
        hold: '#F5A623',         // Amber - HOLD
        buyBg: 'rgba(0,208,124,0.12)',
        sellBg: 'rgba(255,83,83,0.12)',
        holdBg: 'rgba(245,166,35,0.12)',
    },

    // ─── Risk ────────────────────────────────────────────────────
    risk: {
        low: '#00D07C',
        medium: '#F5A623',
        high: '#FF5353',
    },

    // ─── Text ────────────────────────────────────────────────────
    text: {
        primary: '#F0F4FF',
        secondary: '#8B9DC3',
        muted: '#4A5568',
        inverse: '#0A0E1A',
    },

    // ─── Border / Separator ──────────────────────────────────────
    border: {
        default: '#1E2D4A',
        subtle: '#151E2E',
        brand: 'rgba(123,97,255,0.3)',
    },

    // ─── Chart ───────────────────────────────────────────────────
    chart: {
        gradient: ['rgba(123,97,255,0.8)', 'rgba(123,97,255,0)'],
        line: '#7B61FF',
    },
};

export const Spacing = {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
};

export const Radius = {
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
    full: 9999,
};

export const Typography = {
    sizes: {
        xs: 11,
        sm: 13,
        md: 15,
        lg: 17,
        xl: 20,
        xxl: 24,
        xxxl: 30,
        display: 38,
    },
    weights: {
        regular: '400',
        medium: '500',
        semibold: '600',
        bold: '700',
        black: '900',
    },
};

export const Shadow = {
    card: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 12,
        elevation: 8,
    },
    glow: {
        shadowColor: '#7B61FF',
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.4,
        shadowRadius: 16,
        elevation: 10,
    },
};
