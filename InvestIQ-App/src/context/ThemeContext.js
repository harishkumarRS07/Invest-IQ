/**
 * ThemeContext - provides dark/light mode toggle.
 * Reads system preference by default.
 */
import React, { createContext, useContext, useState, useMemo } from 'react';
import { useColorScheme } from 'react-native';
import { Colors } from '../constants/theme';

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
    const systemScheme = useColorScheme();
    const [scheme, setScheme] = useState(systemScheme ?? 'dark');
    const isDark = scheme === 'dark';

    // Memoised so components useMemo([C]) only re-runs on actual theme change
    const C = useMemo(() => (isDark ? Colors : buildLightColors()), [isDark]);

    const toggle = () => setScheme((s) => (s === 'dark' ? 'light' : 'dark'));

    return (
        <ThemeContext.Provider value={{ isDark, colors: C, toggle, scheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

function buildLightColors() {
    return {
        ...Colors,
        bg: {
            primary: '#F0F4FF',
            secondary: '#FFFFFF',
            card: '#FFFFFF',
            elevated: '#E8EEF8',
            input: '#E5EBF5',
        },
        text: {
            primary: '#0A0E1A',
            secondary: '#4A5568',
            muted: '#8B9DC3',
            inverse: '#FFFFFF',
        },
        border: {
            default: '#D1DBF0',
            subtle: '#E5EBF5',
            brand: 'rgba(123,97,255,0.3)',
        },
    };
}

export function useTheme() {
    const ctx = useContext(ThemeContext);
    if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
    return ctx;
}

/** Convenience hook: returns the current resolved color palette. */
export function useColors() {
    return useTheme().colors;
}
