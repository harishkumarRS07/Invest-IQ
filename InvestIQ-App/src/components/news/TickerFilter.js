/**
 * TickerFilter – memoized horizontal ticker chip selector.
 *
 * Performance notes:
 *  - React.memo: skips re-render when tickers, selectedTicker, and onSelectTicker are unchanged
 *  - renderItem stabilized with useCallback (no new function per render)
 */
import React, { useCallback } from 'react';
import { ScrollView, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useColors } from '../../context/ThemeContext';
import { Radius, Spacing, Typography } from '../../constants/theme';

const TickerFilter = React.memo(({ tickers, selectedTicker, onSelectTicker }) => {
    const C = useColors();

    const renderItem = useCallback((ticker) => {
        const isSelected = selectedTicker === ticker;
        const bg = isSelected ? C.brand.purple : C.bg.elevated;
        const color = isSelected ? '#FFFFFF' : C.text.primary;
        const borderColor = isSelected ? C.brand.purple : C.border.default;

        return (
            <TouchableOpacity
                key={ticker}
                style={[styles.chip, { backgroundColor: bg, borderColor }]}
                onPress={() => onSelectTicker(ticker)}
                activeOpacity={0.75}
            >
                <Text style={[styles.text, { color }]}>{ticker}</Text>
            </TouchableOpacity>
        );
    }, [selectedTicker, C, onSelectTicker]);

    return (
        <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.container}
        >
            {tickers.map(renderItem)}
        </ScrollView>
    );
});

export default TickerFilter;

const styles = StyleSheet.create({
    container: {
        paddingHorizontal: Spacing.lg,
        paddingVertical: Spacing.sm,
        gap: Spacing.sm,
    },
    chip: {
        paddingHorizontal: Spacing.md,
        paddingVertical: 8,
        borderRadius: Radius.lg,
        borderWidth: 1,
    },
    text: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.semibold,
    },
});
