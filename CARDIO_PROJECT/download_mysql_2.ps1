$ProgressPreference = 'SilentlyContinue'

$urls = @(
    "https://cdn.mysql.com//Downloads/MySQLInstaller/mysql-installer-web-community-8.0.46.0.msi",
    "https://dev.mysql.com/get/Downloads/MySQLInstaller/mysql-installer-web-community-8.0.46.0.msi",
    "https://cdn.mysql.com//Downloads/MySQLInstaller/mysql-installer-community-8.0.46.0.msi",
    "https://dev.mysql.com/get/Downloads/MySQLInstaller/mysql-installer-community-8.0.46.0.msi",
    "https://dev.mysql.com/downloads/file/?id=552803"
)

$destDir = [System.IO.Path]::Combine($env:USERPROFILE, "Downloads")
$destFile = [System.IO.Path]::Combine($destDir, "mysql-installer-web-community-8.0.46.0.msi")

Write-Host "Target location: $destFile"

$downloaded = $false
foreach ($url in $urls) {
    try {
        Write-Host "Attempting download from: $($url)"
        $client = New-Object System.Net.WebClient
        $client.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        $client.Headers.Add("Referer", "https://dev.mysql.com/downloads/installer/")
        $client.DownloadFile($url, $destFile)

        if (Test-Path $destFile) {
            $size = (Get-Item $destFile).Length
            if ($size -gt 1000000) {
                Write-Host "SUCCESS: Downloaded $([math]::Round($size/1MB, 2)) MB to $destFile"
                $downloaded = $true
                break
            } else {
                Write-Host "File too small ($size bytes). Cleaning..."
                Remove-Item $destFile -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        Write-Host "Download failed for $($url): $($_.Exception.Message)"
    }
}

if (-not $downloaded) {
    Write-Host "Trying curl with referer..."
    & curl.exe -L -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -e "https://dev.mysql.com/downloads/installer/" -o "$destFile" "https://cdn.mysql.com//Downloads/MySQLInstaller/mysql-installer-web-community-8.0.46.0.msi"
    if (Test-Path $destFile) {
        $size = (Get-Item $destFile).Length
        Write-Host "curl size: $size bytes"
    }
}
