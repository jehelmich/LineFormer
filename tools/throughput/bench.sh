#!/bin/bash
# One benchmark config: guard, Windows GPU/CPU sampler, batch_infer.py, merge, compare vs A and C.
# Usage: bench.sh <name> <batch_infer args...>   (env: R = rocm_eval dir, E = lineformer dir, T = output dir)
set -u
NAME=$1; shift
E=${E:?set E to the directory holding ckpt/iter_3000.pth and rocm_eval/}
R=${R:-$E/rocm_eval}
T=${T:-$R/throughput}
PY=/root/lineformer/bin/python
H=/root/LineFormer/tools
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
mkdir -p $T/runs $T/samples $T/compare $T/logs
if ps -eo args | grep -E 'vllm|chandra' | grep -v grep; then echo "ABORT: vllm/chandra running"; exit 3; fi
if ps -eo args | grep -E 'equivalence/(run|profile_forward)\.py.*--device cuda' | grep -v grep; then
  echo "ABORT: tester GPU run active"; exit 4; fi
WT=$(wslpath -w $T)
rm -f $T/samples/$NAME.stop
$PS -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w $H/throughput/gpu_engine_sampler.ps1)" \
    -Out "$WT\\samples\\$NAME.csv" -StopFile "$WT\\samples\\$NAME.stop" > $T/logs/$NAME.sampler.log 2>&1 &
SP=$!
{ echo "=== $(date '+%F %T') $NAME"; uptime; } >> $T/logs/load.log
rm -rf $T/runs/$NAME
cd /root/LineFormer
$PY -B $H/throughput/batch_infer.py --repo /root/LineFormer --config /root/LineFormer/lineformer_swin_t_config.py \
    --ckpt $E/ckpt/iter_3000.pth --images $R/images_wsl.txt --out $T/runs/$NAME --tag $NAME "$@" > $T/logs/$NAME.log 2>&1
rc=$?
{ echo "--- after $NAME rc=$rc"; uptime; } >> $T/logs/load.log
touch $T/samples/$NAME.stop; wait $SP
$PY $H/throughput/merge_samples.py $T/runs/$NAME/run_meta.json $T/samples/$NAME.csv >> $T/logs/$NAME.log 2>&1
tail -n 2 $T/logs/$NAME.log
if [ -f $T/runs/$NAME/run_meta.json ] && grep -q '"status": "done"' $T/runs/$NAME/run_meta.json; then
  $PY -B $H/equivalence/compare.py --ref $R/A --cand $T/runs/$NAME --out $T/compare/A_vs_$NAME.json | head -3
  $PY -B $H/equivalence/compare.py --ref $R/C --cand $T/runs/$NAME --out $T/compare/C_vs_$NAME.json | head -3
fi
exit $rc
