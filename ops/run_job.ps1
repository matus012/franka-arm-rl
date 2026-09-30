# Run one GPU job, sample VRAM + free RAM every 5 s, append a row to ops/gpu_log.csv.
# Usage: powershell -File ops\run_job.ps1 -Job m0_train -LogFile logs\jobs\m0_train.log -PyArgs "scripts\train.py --task ..."  (runs .venv\Scripts\python.exe)
# The job's stdout/stderr go to -LogFile. NaN-guard events are read from the last "NAN_GUARD_EVENTS=<n>" line.
param(
    [Parameter(Mandatory = $true)][string]$Job,
    [Parameter(Mandatory = $true)][string]$LogFile,
    [Parameter(Mandatory = $true)][string]$PyArgs
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
. "$root\scripts\uv_env.ps1"
New-Item -ItemType Directory -Force (Split-Path -Parent $LogFile) | Out-Null
$csv = "$root\ops\gpu_log.csv"
if (-not (Test-Path $csv)) { "job,start,end,minutes,peak_vram_mib,min_free_ram_mib,nan_guard_events,exit_code" | Out-File -Encoding utf8 $csv }

$start = Get-Date
$exe = "$root\.venv\Scripts\python.exe"; $rest = "-u $PyArgs"
$errLog = "$LogFile.err"
$p = Start-Process -FilePath $exe -ArgumentList $rest -NoNewWindow -PassThru -RedirectStandardOutput $LogFile -RedirectStandardError $errLog
$null = $p.Handle  # cache the handle so ExitCode is populated
$peakVram = 0; $minFree = [int]::MaxValue
while (-not $p.HasExited) {
    $v = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    $f = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
    if ($v -gt $peakVram) { $peakVram = $v }
    if ($f -lt $minFree) { $minFree = $f }
    Start-Sleep -Seconds 5
}
$p.WaitForExit()
$end = Get-Date
$mins = [math]::Round(($end - $start).TotalMinutes, 2)
$nan = ""
$hit = Select-String -Path $LogFile, $errLog -Pattern "NAN_GUARD_EVENTS=(\d+)" -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($hit) { $nan = $hit.Matches[0].Groups[1].Value }
"$Job,$($start.ToString('s')),$($end.ToString('s')),$mins,$peakVram,$minFree,$nan,$($p.ExitCode)" | Out-File -Append -Encoding utf8 $csv
Write-Output "[run_job] $Job exit=$($p.ExitCode) minutes=$mins peakVRAM=${peakVram}MiB minFreeRAM=${minFree}MiB nan=$nan"
exit $p.ExitCode

