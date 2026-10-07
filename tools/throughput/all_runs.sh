#!/bin/bash
# The throughput matrix of ROCM_PLAN.md step 4b (E1-E3), one config after the other.
# Usage: all_runs.sh [names...]  (default: all). Output: $T/runs/<name>, logs, compare/.
B=/root/LineFormer/tools/throughput/bench.sh
G="--device cuda:0 --msda pytorch --repeat ${REPEAT:-5}"
declare -A CFG=(
  [base_getds]="--mode getds"
  [serial]="--mode serial"
  [e1_n1]="--mode pipeline --gpu-workers 1"
  [e2_n2]="--mode pipeline --gpu-workers 2"
  [e2_n3]="--mode pipeline --gpu-workers 3"
  [e2_n4]="--mode pipeline --gpu-workers 4"
  [e3_b2]="--mode pipeline --gpu-workers 1 --batch 2"
  [e3_b4]="--mode pipeline --gpu-workers 1 --batch 4"
  [e3_b8]="--mode pipeline --gpu-workers 1 --batch 8"
)
ORDER=(base_getds serial e1_n1 e2_n2 e2_n3 e2_n4 e3_b2 e3_b4 e3_b8)
names=("$@"); [ ${#names[@]} -eq 0 ] && names=("${ORDER[@]}")
for n in "${names[@]}"; do
  args=${CFG[$n]:-$EXTRA}
  echo "=== $(date '+%F %T') $n: $args"
  bash $B $n $G $args
  rc=$?
  echo "--- $n rc=$rc"
  [ $rc -eq 3 ] && { echo "vllm/chandra seen: stopping"; exit 3; }
done
