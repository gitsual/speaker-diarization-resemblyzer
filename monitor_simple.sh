#!/bin/bash
clear
echo "============================================================"
echo "📊 Monitor - Mapeo Simple SRT"
echo "============================================================"
echo ""

if ps aux | grep -q "[p]ython3 9_simple_srt_mapping"; then
    PID=$(ps aux | grep "[p]ython3 9_simple_srt_mapping" | awk '{print $2}')
    CPU=$(ps aux | grep "[p]ython3 9_simple_srt_mapping" | awk '{print $3}')
    MEM=$(ps aux | grep "[p]ython3 9_simple_srt_mapping" | awk '{print $4}')
    echo "✅ Proceso activo (PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%)"
else
    echo "❌ Proceso terminado"
fi

echo ""
echo "============================================================"
echo "📄 Progreso actual:"
echo "============================================================"
tail -20 simple_mapping_final.log | grep -E "(Procesando|INFO|muestras|completada|RESUMEN)" || tail -20 simple_mapping_final.log

echo ""
echo "============================================================"
echo "📊 Archivos generados:"
echo "============================================================"
ls -lh sesion_*/sesion_*_diarizada.txt 2>/dev/null || echo "Aún procesando..."

echo ""
echo "Comando: ./monitor_simple.sh"
echo "Ver log completo: tail -f simple_mapping_final.log"
