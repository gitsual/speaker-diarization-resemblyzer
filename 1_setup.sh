#!/bin/bash
# Script de instalación para sistema de diarización con enrollment
# Compatible con RTX 4080 SUPER + CUDA 12.x

set -e

echo "=== Instalación de Sistema de Diarización con Speaker Enrollment ==="
echo ""

# Verificar CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi no encontrado. Verifica tu instalación de CUDA."
    exit 1
fi

echo "GPU detectada:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv_diarization
source venv_diarization/bin/activate

# Actualizar pip
pip install --upgrade pip setuptools wheel

# PyTorch con CUDA 12.1
echo ""
echo "Instalando PyTorch con soporte CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyAnnote Audio y dependencias principales
echo ""
echo "Instalando PyAnnote Audio 3.x..."
pip install pyannote.audio==3.1.1

# Dependencias adicionales
echo ""
echo "Instalando dependencias adicionales..."
pip install \
    pyannote.core \
    pyannote.metrics \
    pydub \
    soundfile \
    librosa \
    gradio \
    plotly \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    tqdm

# Instalar ffmpeg si no existe (necesario para procesar .m4a)
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "ffmpeg no encontrado. Instalando..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    else
        echo "Por favor instala ffmpeg manualmente: https://ffmpeg.org/download.html"
    fi
fi

# Crear estructura de directorios
echo ""
echo "Creando estructura de directorios..."
mkdir -p speaker_profiles
mkdir -p temp_audio_segments
mkdir -p output_diarized
mkdir -p logs

echo ""
echo "=== Instalación completada ==="
echo ""
echo "Siguiente paso: Obtén tu HuggingFace token en https://huggingface.co/settings/tokens"
echo "Acepta las condiciones de uso de pyannote/speaker-diarization-3.1:"
echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "  https://huggingface.co/pyannote/segmentation-3.0"
echo ""
echo "Luego ejecuta: export HF_TOKEN='tu_token_aqui'"
echo "O guárdalo en un archivo .env"
echo ""
echo "Para activar el entorno: source venv_diarization/bin/activate"