/**
 * NewsCard – memoized news article card.
 *
 * Performance notes:
 *  - React.memo with custom comparator: skips re-render when article identity unchanged
 *  - handlePress stable via useCallback
 *  - formatTimeAgo called via useMemo so it's only recalculated when timestamp changes
 */
import React, { useCallback, useMemo } from 'react';
import { View, Text, StyleSheet, Linking, TouchableOpacity } from 'react-native';
import { useColors } from '../../context/ThemeContext';
import { Radius, Spacing, Typography } from '../../constants/theme';
import SentimentBadge from './SentimentBadge';

// Helper function to format timestamp – pure, defined outside component
const formatTimeAgo = (timestamp) => {
    if (!timestamp) return '';
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        return `${diffDays}d ago`;
    } catch {
        return '';
    }
};

/**
 * Custom equality check: only re-render if the article reference changed.
 * Avoids renders when parent re-renders but article data is the same.
 */
const arePropsEqual = (prev, next) => {
    return (
        prev.article.title === next.article.title &&
        prev.article.link === next.article.link &&
        prev.article.publisher === next.article.publisher &&
        prev.article.providerPublishTime === next.article.providerPublishTime &&
        prev.article.sentiment_label === next.article.sentiment_label &&
        prev.article.sentiment_score === next.article.sentiment_score
    );
};

const NewsCard = React.memo(({ article }) => {
    const C = useColors();

    const handlePress = useCallback(() => {
        if (article.link) {
            Linking.openURL(article.link).catch((err) =>
                console.error("Couldn't load page", err)
            );
        }
    }, [article.link]);

    const timeAgo = useMemo(
        () => formatTimeAgo(article.providerPublishTime),
        [article.providerPublishTime]
    );

    return (
        <TouchableOpacity
            activeOpacity={0.8}
            onPress={handlePress}
            style={[styles.card, { backgroundColor: C.bg.elevated, borderColor: C.border.default }]}
        >
            <View style={styles.headerRow}>
                <Text style={[styles.publisher, { color: C.text.secondary }]}>{article.publisher}</Text>
                <Text style={[styles.timestamp, { color: C.text.muted }]}>{timeAgo}</Text>
            </View>
            <Text style={[styles.title, { color: C.text.primary }]} numberOfLines={3}>
                {article.title}
            </Text>
            <View style={styles.footerRow}>
                {article.sentiment_label && (
                    <SentimentBadge sentiment={article.sentiment_label} score={article.sentiment_score} />
                )}
            </View>
        </TouchableOpacity>
    );
}, arePropsEqual);

export default NewsCard;

const styles = StyleSheet.create({
    card: {
        padding: Spacing.md,
        borderRadius: Radius.md,
        borderWidth: 1,
        marginBottom: Spacing.md,
    },
    headerRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.xs,
    },
    publisher: {
        fontSize: Typography.sizes.sm,
        fontWeight: Typography.weights.bold,
    },
    timestamp: {
        fontSize: Typography.sizes.xs,
    },
    title: {
        fontSize: Typography.sizes.md,
        fontWeight: Typography.weights.semibold,
        lineHeight: 22,
        marginBottom: Spacing.sm,
    },
    footerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: Spacing.xs,
    },
});
