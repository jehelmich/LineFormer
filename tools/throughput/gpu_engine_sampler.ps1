# Windows-side sampler for WSL GPU work (rocm-smi does not work in WSL; the ROCm profiler sees no device events).
# Every ~1 s: "\GPU Engine(*)\Utilization Percentage" summed over processes per engine (luid, phys, eng, engtype),
# plus "\Processor(_Total)\% Processor Time" (whole Windows host, which includes the WSL VM).
# CSV lines: unix_time,key,value   key = engine "luid_..._phys_N_eng_M_engtype_X" or "host_cpu".
# Runs until the stop file exists.
param([string]$Out, [string]$StopFile)
"unix_time,key,value" | Out-File -FilePath $Out -Encoding ascii
while (-not (Test-Path $StopFile)) {
    try {
        $s = Get-Counter -Counter @('\GPU Engine(*)\Utilization Percentage', '\Processor(_Total)\% Processor Time') -ErrorAction Stop
    } catch { Start-Sleep -Milliseconds 500; continue }
    $u = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $agg = @{}
    foreach ($c in $s.CounterSamples) {
        if ($c.Path -like '*processor(_total)*') { $agg['host_cpu'] = $c.CookedValue; continue }
        $k = $c.InstanceName -replace '^pid_\d+_', ''
        if ($agg.ContainsKey($k)) { $agg[$k] += $c.CookedValue } else { $agg[$k] = $c.CookedValue }
    }
    $lines = foreach ($k in $agg.Keys) { "{0},{1},{2}" -f $u, $k, $agg[$k] }
    $lines | Out-File -FilePath $Out -Append -Encoding ascii
}
