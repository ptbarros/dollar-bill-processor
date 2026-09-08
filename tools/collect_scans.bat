@echo off
rem ===================================================================
rem  Dollar Detective - Scan Data Collector
rem
rem  Drop this file into a MONTH folder (e.g. "August Scans") that
rem  contains the strap subfolders (801, 802, 903, ...) and double-click.
rem
rem  It gathers the small report/label files from every subfolder,
rem  builds a manifest of all files (so cropped serials are captured
rem  WITHOUT copying the big images), and packages everything into a
rem  single .zip sitting right next to this .bat.
rem
rem  Nothing is deleted or changed - it only READS and makes one .zip.
rem ===================================================================
setlocal EnableExtensions
set "BATFILE=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText($env:BATFILE);$i=$s.LastIndexOf('#PSBODY#');iex $s.Substring($i+8)"
echo.
pause
exit /b

#PSBODY#
$ErrorActionPreference = 'Stop'
try {
    $root  = Split-Path -Parent $env:BATFILE
    $leaf  = Split-Path -Leaf  $root
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stage = Join-Path $env:TEMP ("dbp_collect_" + $stamp)
    New-Item -ItemType Directory -Force -Path $stage | Out-Null

    Write-Host ""
    Write-Host ("Collecting from: {0}" -f $root)
    Write-Host "Scanning subfolders..."

    # 1) Copy the small report + label files, preserving the strap subfolder
    $patterns = 'results_*.csv','summary_*.txt','non_fancy_files_*.txt',
                'review_queue_*.json','*.pdf','*.docx','*.xlsx'
    $copied = 0
    foreach ($pat in $patterns) {
        Get-ChildItem -Path $root -Recurse -File -Filter $pat -ErrorAction SilentlyContinue |
            ForEach-Object {
                $rel  = $_.FullName.Substring($root.Length).TrimStart('\')
                $dest = Join-Path $stage $rel
                New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
                Copy-Item $_.FullName $dest -Force
                $copied++
            }
    }
    Write-Host ("  Copied {0} report/label files." -f $copied)

    # 2) Manifest of EVERY file (crop filenames = kept serials; no images copied)
    $manifest = Join-Path $stage '_all_files_manifest.csv'
    $rootLen  = $root.Length
    Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            [PSCustomObject]@{
                RelPath  = $_.FullName.Substring($rootLen).TrimStart('\')
                Folder   = (Split-Path $_.FullName -Parent).Substring($rootLen).TrimStart('\')
                Name     = $_.Name
                Ext      = $_.Extension
                SizeKB   = [math]::Round($_.Length / 1KB, 1)
                Modified = $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
            }
        } | Export-Csv -Path $manifest -NoTypeInformation -Encoding UTF8
    $fileCount = (Import-Csv $manifest | Measure-Object).Count
    Write-Host ("  Indexed {0} total files into manifest." -f $fileCount)

    # 3) Zip it up next to this .bat
    $safe = ($leaf -replace '[^\w\-]', '_')
    $zip  = Join-Path $root ("DBP_data_" + $safe + "_" + $stamp + ".zip")
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -Force
    Remove-Item $stage -Recurse -Force

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host " DONE." -ForegroundColor Green
    Write-Host (" ZIP created: {0}" -f (Split-Path $zip -Leaf)) -ForegroundColor Green
    Write-Host (" Location:    {0}" -f $root)
    Write-Host " Email/send that .zip to Paul." -ForegroundColor Green
    Write-Host "==================================================" -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host ("ERROR: {0}" -f $_.Exception.Message) -ForegroundColor Red
    Write-Host "Nothing was changed. Send Paul a screenshot of this window."
}
