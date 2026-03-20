/**
 * SentimentBadge – memoized badge for news sentiment.
 * Wrapped in React.memo to avoid re-rendering when parent re-renders
 * with the same sentiment/score props.
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Radius, Spacing, Typography } from '../../constants/theme';
import { useColors } from '../../context/ThemeContext';

const SentimentBadge = React.memo(({ sentiment, score }) => {
    const C = useColors();

    let bg = C.signal.holdBg;
    let color = C.signal.hold;

    if (sentiment === 'Positive') {
        bg = C.signal.buyBg;
        color = C.signal.buy;
    } else if (sentiment === 'Negative') {
        bg = C.signal.sellBg;
        color = C.signal.sell;
    }

    return (
        <View style={[styles.badge, { backgroundColor: bg, borderColor: color + '40' }]}>
            <Text style={[styles.text, { color }]}>
                {sentiment} {score ? `(${score.toFixed(2)})` : ''}
            </Text>
        </View>
    );
});

export default SentimentBadge;

const styles = StyleSheet.create({
    badge: {
        paddingHorizontal: Spacing.sm,
        paddingVertical: 4,
        borderRadius: Radius.sm,
        borderWidth: 1,
    },
    text: {
        fontSize: Typography.sizes.xs,
        fontWeight: Typography.weights.bold,
    },
});
