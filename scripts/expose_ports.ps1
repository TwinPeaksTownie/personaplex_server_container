# Get the IP address of the PersonaPlex-WSL distro
$wslIp = wsl -d PersonaPlex-WSL hostname -I
$wslIp = $wslIp.Trim()

if (-not $wslIp) {
    Write-Error "Could not detect IP address for PersonaPlex-WSL. Is the distro running?"
    exit 1
}

Write-Host "Found PersonaPlex-WSL IP: $wslIp"

# Ports to expose
$ports = @(5173, 8080)

foreach ($port in $ports) {
    # 1. Remove existing proxy (cleanup)
    Write-Host "Cleaning up existing rules for port $port..."
    netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 | Out-Null

    # 2. Add new proxy
    Write-Host "Forwarding Port $port -> $wslIp:$port"
    netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp

    # 3. firewall rule
    $ruleName = "PersonaPlex_Port_$port"
    Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    Write-Host "Opening Firewall Port $port..."
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -LocalPort $port -Protocol TCP -Action Allow | Out-Null
}

Write-Host "✅ Network Configuration Complete!"
Write-Host "You can now access PersonaPlex from other devices at http://$((Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi','Ethernet' | Select-Object -ExpandProperty IPAddress | Select-Object -First 1)):5173"
