#!/bin/zsh
# Joint-model validation (v3) in parallel: one process per part, then the summary.
# Run from the repository root:  caffeinate -i zsh bn/run_v3.sh
# Parts already written to transport_output/jv_parts_v3/ are skipped, so the
# script can be started again after an interruption.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs

python -u -m bn.validate_issp --tag _v3 --networks-only

python -u -m bn.validate_issp --tag _v3 --scenarios same > logs/v3_same.log 2>&1 &
for o in pay_higher_prices pay_higher_taxes cut_living_standard do_right_despite_cost avoid_products; do
  python -u -m bn.validate_issp --tag _v3 --scenarios small_cross --outcomes $o > logs/v3_small_cross_$o.log 2>&1 &
done
echo "6 processes running; progress: tail -f logs/v3_*.log"
wait

python -u -m bn.validate_issp --tag _v3 --summarise-only | tee logs/v3_summary.log
