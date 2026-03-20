/**
 * Tab Layout
 * Standard Expo tabs to respect Safe Areas and System Buttons.
 */
import { Tabs } from 'expo-router';
import { useColors } from '../../src/context/ThemeContext';

import { Ionicons } from '@expo/vector-icons';
import { Platform } from 'react-native';

export default function TabLayout() {
    const C = useColors();

    return (
        <Tabs
            screenOptions={{
                headerShown: false,
                tabBarStyle: {
                    backgroundColor: C.bg.elevated,
                    borderTopColor: C.border.default,
                },
                tabBarItemStyle: {
                    justifyContent: 'center',
                    alignItems: 'center',
                },
                tabBarLabelStyle: {
                    fontSize: 12,
                    fontWeight: '600',
                    marginTop: 4,
                },
                tabBarActiveTintColor: C.brand.purple,
                tabBarInactiveTintColor: C.text.muted,
            }}
        >
            <Tabs.Screen
                name="dashboard"
                options={{
                    title: 'Dashboard',
                    tabBarIcon: ({ color, size }) => (
                        <Ionicons name="analytics" size={size} color={color} />
                    ),
                    tabBarLabel: 'Dashboard'
                }}
            />
            <Tabs.Screen
                name="news"
                options={{
                    title: 'News',
                    tabBarIcon: ({ color, size }) => (
                        <Ionicons name="newspaper" size={size} color={color} />
                    ),
                    tabBarLabel: 'News'
                }}
            />
            <Tabs.Screen
                name="portfolio"
                options={{
                    title: 'Portfolio',
                    tabBarIcon: ({ color, size }) => (
                        <Ionicons name="pie-chart" size={size} color={color} />
                    ),
                    tabBarLabel: 'Portfolio'
                }}
            />
            <Tabs.Screen
                name="settings"
                options={{
                    title: 'Settings',
                    tabBarIcon: ({ color, size }) => (
                        <Ionicons name="settings" size={size} color={color} />
                    ),
                    tabBarLabel: 'Settings'
                }}
            />
        </Tabs>
    );
}
