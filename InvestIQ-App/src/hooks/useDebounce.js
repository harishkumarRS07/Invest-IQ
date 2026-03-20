/**
 * useDebounce – returns a debounced value that updates only after
 * the specified delay has passed without new changes.
 *
 * Usage:
 *   const debouncedSearch = useDebounce(search, 300);
 */
import { useState, useEffect } from 'react';

export function useDebounce(value, delay = 300) {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const timer = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => clearTimeout(timer);
    }, [value, delay]);

    return debouncedValue;
}
