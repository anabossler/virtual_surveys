#!/bin/bash
# Analisis de sensibilidad al numero de bins (K)
# Ejecuta el pipeline completo para K en {3, 5, 7}
# Uso: bash run_k_sensitivity_pipeline.sh

set -e

PASO1="step1.py"
PASO2="step2.py"
PASO3="step3.py"
PASO4="step4.py"
PASO5="step5.py"
PASO6="step6.py"
PASO7="step7.py"
PASO8="step8.py"
DIR_RESULTADOS="k_sensitivity_results"

mkdir -p "$DIR_RESULTADOS"

cp "$PASO1" "${PASO1}.respaldo"

for K in 3 5 7; do
    echo ""
    echo "  K = $K bins"
    echo ""

    sed -i.bak \
        "s/n_bins=[0-9][0-9]*/n_bins=$K/g" \
        "$PASO1"

    echo "  [1/8] step1.py — DAG structure learning (K=$K)..."
    python3 "$PASO1" 2>&1 | grep -E "Variables:|DAG:|edges" | tail -5

    echo "  [2/8] step2.py — Prior construction from literature..."
    python3 "$PASO2" 2>&1 | grep -E "prior|lambda|ERROR" | tail -5

    echo "  [3/8] step3.py — Bayesian fusion..."
    python3 "$PASO3" 2>&1 | grep -E "CPT|nodes|fused|ERROR" | tail -5

    echo "  [4/8] step4.py — Synthetic population generation..."
    python3 "$PASO4" 2>&1 | grep -E "synthetic|rows|ERROR" | tail -5

    echo "  [5/8] step5.py — Validation (3-level framework)..."
    python3 "$PASO5" 2>&1 | grep -E "level|pass|ERROR" | tail -5

    echo "  [6/8] step6.py — Variable-level diagnostics (TVD)..."
    python3 "$PASO6" 2>&1 | grep -E "TVD|mean|ERROR" | tail -5

    echo "  [7/8] step7.py — Baseline comparison..."
    python3 "$PASO7" 2>&1 | grep -E "comparison|ERROR" | tail -5

    echo "  [8/8] step8.py — External validation (Ipsos)..."
    python3 "$PASO8" 2>&1 | grep -E "ipsos|validation|ERROR" | tail -5

    cp "step6_output/tvd_per_variable_fixed.csv" \
       "$DIR_RESULTADOS/tvd_K${K}.csv"
    echo "  Guardado: $DIR_RESULTADOS/tvd_K${K}.csv"
done

cp "${PASO1}.respaldo" "$PASO1"
echo ""
echo "  step1.py restaurado a K=5"

echo ""
echo "  RESUMEN DE RESULTADOS"
echo ""
python3 - << 'FIN_PYTHON'
import pandas as pd
import numpy as np
import os

directorio = "k_sensitivity_results"
filas = []

for K in [3, 5, 7]:
    ruta = os.path.join(directorio, f"tvd_K{K}.csv")
    if not os.path.exists(ruta):
        print(f"  Archivo no encontrado: {ruta}")
        continue
    df = pd.read_csv(ruta)
    col_tvd = 'BVS (S1)' if 'BVS (S1)' in df.columns else df.columns[1]
    bvs = df[col_tvd].dropna()
    filas.append({
        'K': K,
        'n_vars': len(bvs),
        'tvd_medio': round(bvs.mean(), 4),
        'tvd_mediana': round(bvs.median(), 4),
        'pct_excelente': round((bvs < 0.05).mean() * 100, 1),
        'pct_menor_10': round((bvs < 0.10).mean() * 100, 1),
        'sobre_15': int((bvs > 0.15).sum()),
    })

resumen = pd.DataFrame(filas)
print(resumen.to_string(index=False))
resumen.to_csv(os.path.join(directorio, "resumen.csv"), index=False)

base = resumen[resumen['K'] == 5]
if not base.empty:
    tvd_base = base['tvd_medio'].values[0]
    exc_base  = base['pct_excelente'].values[0]
    print("\n  Diferencia respecto a K=5:")
    for _, fila in resumen.iterrows():
        if fila['K'] != 5:
            print(f"    K={int(fila['K'])}: TVD medio {(fila['tvd_medio'] - tvd_base) * 100:+.1f}pp, "
                  f"excelente {fila['pct_excelente'] - exc_base:+.1f}pp")

print("\n  Resumen guardado en: k_sensitivity_results/resumen.csv")
FIN_PYTHON
