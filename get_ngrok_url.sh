#!/bin/bash
# Script to extract and update ngrok URL in frontend config
# Makes API calls work through ngrok tunnel

set -e

CONFIG_FILE="InvestIQ-App/config/api.js"
NGROK_API="http://127.0.0.1:4040/api/tunnels"

echo "=========================================="
echo "ngrok URL Extractor & Frontend Updater"
echo "=========================================="
echo ""

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Fetch ngrok tunnel status
echo "Fetching ngrok tunnel status..."
RESPONSE=$(curl -s "$NGROK_API" 2>/dev/null || echo "{}")

# Extract HTTPS URL (ngrok always provides HTTPS as primary)
if command -v jq &> /dev/null; then
    NGROK_URL=$(echo "$RESPONSE" | jq -r '.tunnels[] | select(.proto=="https") | .public_url' | head -1)
else
    NGROK_URL=$(echo "$RESPONSE" | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4)
fi

if [ -z "$NGROK_URL" ] || [ "$NGROK_URL" == "null" ]; then
    echo "❌ Could not get ngrok URL. Make sure:"
    echo "   1. ngrok is installed: https://ngrok.com/download"
    echo "   2. Tunnel is running: ngrok http 8000"
    echo "   3. ngrok API is accessible at $NGROK_API"
    exit 1
fi

echo "✅ ngrok Tunnel Active"
echo "   Public URL: $NGROK_URL"
echo ""

# Create API URL with /api/v1 path
API_URL="${NGROK_URL}/api/v1"
echo "API URL: $API_URL"
echo ""

# Update config file based on OS
echo "Updating $CONFIG_FILE..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - requires empty string after -i
    sed -i '' "s|export const API_NGROK_URL = '[^']*';|export const API_NGROK_URL = '$API_URL';|g" "$CONFIG_FILE"
else
    # Linux
    sed -i "s|export const API_NGROK_URL = '[^']*';|export const API_NGROK_URL = '$API_URL';|g" "$CONFIG_FILE"
fi

echo "✅ Frontend config updated successfully!"
echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo "1. Restart Expo:"
echo "   cd InvestIQ-App"
echo "   npx expo start --tunnel -c"
echo ""
echo "2. Test login on your mobile device"
echo ""
echo "3. Monitor ngrok traffic:"
echo "   Open http://127.0.0.1:4040 in browser"
echo "=========================================="
