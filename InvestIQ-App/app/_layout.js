/**
 * Root Expo Router layout.
 * Wraps the app in Context providers and handles auth-based navigation.
 */
import { useState, useEffect } from 'react';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View, StyleSheet } from 'react-native';
import { AuthProvider, useAuth } from '../src/context/AuthContext';
import { ThemeProvider } from '../src/context/ThemeContext';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { Colors } from '../src/constants/theme';
import AnimatedSplash from '../src/components/SplashScreen';

// ─── Auth guard: redirect unauthenticated users to /login ──────────────────
function AuthGuard({ children }) {
    const { isAuthenticated, loading } = useAuth();
    const segments = useSegments();
    const router = useRouter();

    useEffect(() => {
        if (loading) return;
        const inAuthGroup = segments[0] === '(auth)';
        if (!isAuthenticated && !inAuthGroup) {
            router.replace('/(auth)/login');
        } else if (isAuthenticated && inAuthGroup) {
            router.replace('/(tabs)/dashboard');
        }
    }, [isAuthenticated, loading, segments]);

    if (loading) {
        return <View style={{ flex: 1, backgroundColor: Colors.bg.primary }} />;
    }

    return children;
}

export default function RootLayout() {
    const [splashComplete, setSplashComplete] = useState(false);

    return (
        <GestureHandlerRootView style={styles.root}>
            <ThemeProvider>
                <AuthProvider>
                    <StatusBar style="light" />
                    {!splashComplete && (
                        <AnimatedSplash onFinish={() => setSplashComplete(true)} />
                    )}
                    {splashComplete && (
                        <AuthGuard>
                            <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: Colors.bg.primary } }}>
                                <Stack.Screen name="(auth)" />
                                <Stack.Screen name="(tabs)" />
                                <Stack.Screen
                                    name="stock/[symbol]"
                                    options={{
                                        headerShown: false,
                                        presentation: 'card',
                                        animation: 'slide_from_right',
                                    }}
                                />
                            </Stack>
                        </AuthGuard>
                    )}
                </AuthProvider>
            </ThemeProvider>
        </GestureHandlerRootView>
    );
}

const styles = StyleSheet.create({
    root: { flex: 1, backgroundColor: Colors.bg.primary },
});
