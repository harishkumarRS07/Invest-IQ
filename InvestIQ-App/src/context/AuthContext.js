/**
 * AuthContext - InvestIQ
 * Manages JWT auth state globally using React Context API.
 * Persists token in Expo SecureStore between sessions.
 */
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authApi, saveToken, getToken, removeToken } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);   // { email, name }
    const [token, setToken] = useState(null);
    const [loading, setLoading] = useState(true);    // true while restoring session

    // ─── Restore session on mount ────────────────────────────────────────────
    useEffect(() => {
        (async () => {
            try {
                const savedToken = await getToken();
                if (savedToken) {
                    setToken(savedToken);
                    const me = await authApi.me();
                    setUser({ email: me.email, name: me.name });
                }
            } catch {
                // Token expired or invalid – clear it
                await removeToken();
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    // ─── Register ────────────────────────────────────────────────────────────
    const register = useCallback(async (email, password, name) => {
        const data = await authApi.register(email, password, name);
        await saveToken(data.token);
        setToken(data.token);
        setUser({ email: data.email, name: data.name });
        return data;
    }, []);

    // ─── Login ───────────────────────────────────────────────────────────────
    const login = useCallback(async (email, password) => {
        const data = await authApi.login(email, password);
        await saveToken(data.token);
        setToken(data.token);
        setUser({ email: data.email, name: data.name });
        return data;
    }, []);

    // ─── Logout ─────────────────────────────────────────────────────────────
    const logout = useCallback(async () => {
        await removeToken();
        setToken(null);
        setUser(null);
    }, []);

    return (
        <AuthContext.Provider value={{ user, token, loading, login, register, logout, isAuthenticated: !!user }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within AuthProvider');
    return ctx;
}
