
# This script dynamically sets the correct Wi-Fi IP address preventing Expo 
# from clinging to VMware/Virtual Box virtual network adapters.

Write-Host "Detecting active Wi-Fi IP Address..." -ForegroundColor Cyan

# Fetch the IPv4 address of the Wi-Fi adapter
$wifiIp = (Get-NetIPAddress -InterfaceAlias '*Wi-Fi*' -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress

if ([string]::IsNullOrWhiteSpace($wifiIp)) {
    Write-Host "Warning: Could not automatically detect a Wi-Fi adapter IP. Defaulting to standard Expo lookup." -ForegroundColor Yellow
} else {
    Write-Host "Found Wi-Fi Network IP: $wifiIp" -ForegroundColor Green
    $env:REACT_NATIVE_PACKAGER_HOSTNAME = $wifiIp
}

Write-Host "Starting Expo Local Server..." -ForegroundColor Cyan
npx expo start --clear
