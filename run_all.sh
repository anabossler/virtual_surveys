#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper.
#
# Inputs (not redistributed; see README):
#   metadata/3391_num.csv        CIS Study 3391 microdata (www.cis.es)
#   metadata/ZA7650_v2-0-0.dta   ISSP 2020 Environment IV (GESIS, doi:10.4232/1.14153)
#
# Usage:
#   bash run_all.sh                 main run (seed 42)
#   bash run_all.sh --seeds 7 2024  main run plus structure-variability runs
#   bash run_all.sh --paper         also assemble paper/main.tex from the outputs
#
# One log per stage in logs/; the script stops at the first failing stage.

set -euo pipefail
cd "$(dirname "$0")"

SEEDS=()
PAPER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --paper) PAPER=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

CIS=metadata/3391_num.csv
ISSP=metadata/ZA7650_v2-0-0.dta
[ -f "$CIS" ] || { echo "missing $CIS (download from www.cis.es)" >&2; exit 1; }
mkdir -p logs

stage() {
  local name="$1"; shift
  local start; start=$(date +%s)
  if "$@" > "logs/$name.log" 2>&1; then
    echo "$name ok $(( $(date +%s) - start ))s"
  else
    echo "$name FAILED (see logs/$name.log)" >&2
    exit 1
  fi
}

export BVS_SEED="${BVS_SEED:-42}"

# Observed variables: structure and population from national data alone
stage step1            python step1.py
stage step2            python step2.py
stage evidence_table   python evidence_table.py
stage step4            python step4.py
stage missingness      python missingness.py

# Unobserved outcome: transfer model and measurement plan
stage transfer_model   python transfer_model.py --cis "$CIS"
stage measurement_plan python measurement_plan.py --cis "$CIS"
stage step4b           python step4b_targets.py

# Validation
stage step6            python step6.py
stage step9            python step9.py

# Cross-national transport error (requires the ISSP file)
if [ -f "$ISSP" ]; then
  stage transport_issp     python transport_issp.py --issp "$ISSP" --out transport_output
  stage transfer_transport python transfer_transport.py --cis "$CIS"
else
  echo "transport_issp skipped: $ISSP not found (download from GESIS)" >&2
fi

# Structure variability across seeds
if [ ${#SEEDS[@]} -gt 0 ]; then
  runs=("42=.:.")
  for seed in "${SEEDS[@]}"; do
    dir="runs/seed_$seed"
    mkdir -p "$dir/metadata"
    cp ./*.py ./*.csv "$dir/"
    ln -sf "$(pwd)/$CIS" "$dir/$CIS"
    ( cd "$dir" && export BVS_SEED="$seed" \
      && python step1.py > step1.log 2>&1 && python step2.py > step2.log 2>&1 \
      && python step4.py > step4.log 2>&1 && python step4b_targets.py > step4b.log 2>&1 \
      && python step6.py > step6.log 2>&1 ) || { echo "seed $seed FAILED (see $dir)" >&2; exit 1; }
    echo "seed_$seed ok"
    runs+=("$seed=$dir:$dir")
  done
  stage structure_variability python structure_variability.py --out structure_variability.json "${runs[@]}"
fi

# Figures
stage figures python make_figures.py --base . --out figures

# Manuscript (fills every table and number from the outputs above)
if [ "$PAPER" = 1 ]; then
  stage paper python paper/assemble.py paper/parts paper/deposit paper/audit paper/main.tex transfer_output .
  cp figures/*.png paper/
  echo "paper/main.tex written; compile with: cd paper && pdflatex main.tex (three times)"
fi

echo "done"
