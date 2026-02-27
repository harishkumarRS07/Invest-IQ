/**
 * Tab Layout
 * Uses tabBar prop so FloatingIQMenu receives proper navigation + state props.
 */
import { Tabs } from 'expo-router';
import FloatingIQMenu from '../../src/components/FloatingIQMenu';

export default function TabLayout() {
    return (
        <Tabs
            tabBar={(props) => <FloatingIQMenu {...props} />}
            screenOptions={{
                headerShown: false,
                tabBarStyle: { display: 'none' },  // hide native bar
            }}
        >
            <Tabs.Screen name="dashboard" />
            <Tabs.Screen name="portfolio" />
            <Tabs.Screen name="settings" />
        </Tabs>
    );
}
