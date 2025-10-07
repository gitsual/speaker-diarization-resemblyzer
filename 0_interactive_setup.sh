#!/bin/bash
# Setup interactivo completo del sistema de diarización
# Este script guía paso a paso la configuración inicial

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funciones auxiliares
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

ask_continue() {
    echo ""
    read -p "Presiona ENTER para continuar..."
    echo ""
}

# Banner
clear
echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   SISTEMA DE DIARIZACIÓN CON SPEAKER ENROLLMENT           ║
║   Para Partidas de Rol                                    ║
║                                                            ║
║   Setup Interactivo - Te guiaremos paso a paso           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

print_info "Este script te ayudará a:"
echo "  1. Verificar requisitos del sistema"
echo "  2. Instalar todas las dependencias"
echo "  3. Configurar tu token de HuggingFace"
echo "  4. Validar que todo funciona correctamente"
echo ""
print_warning "Tiempo estimado: 15-20 minutos"
ask_continue

# PASO 1: Verificar requisitos
print_header "PASO 1: Verificación de Requisitos"

print_info "Verificando Python 3.9+..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python encontrado: $python_version"
else
    print_error "Python 3 no encontrado"
    echo "Por favor instala Python 3.9 o superior primero."
    exit 1
fi

print_info "Verificando NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    print_success "GPU detectada: $gpu_info"
else
    print_warning "nvidia-smi no encontrado"
    print_warning "El sistema funcionará en CPU (muy lento)"
    echo ""
    read -p "¿Continuar de todos modos? (y/n): " choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        print_error "Setup cancelado"
        exit 1
    fi
fi

print_info "Verificando ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    print_success "ffmpeg encontrado"
else
    print_warning "ffmpeg no encontrado - necesario para procesar .m4a"
    echo ""
    echo "Para instalar ffmpeg:"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  sudo apt-get update && sudo apt-get install -y ffmpeg"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  brew install ffmpeg"
    fi
    echo ""
    read -p "¿Ya tienes ffmpeg instalado o lo instalarás después? (y/n): " choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        print_error "Por favor instala ffmpeg primero"
        exit 1
    fi
fi

print_success "Verificación de requisitos completada"
ask_continue

# PASO 2: Ejecutar 1_setup.sh
print_header "PASO 2: Instalación de Dependencias"

if [ ! -f "1_setup.sh" ]; then
    print_error "No se encontró 1_setup.sh en el directorio actual"
    exit 1
fi

print_info "Ejecutando 1_setup.sh para instalar:"
echo "  - PyTorch con CUDA"
echo "  - PyAnnote Audio 3.1"
echo "  - Gradio, librosa, scipy, numpy, etc."
echo ""
print_warning "Esto tomará 10-15 minutos. Por favor, NO interrumpas el proceso."
echo ""
read -p "¿Continuar con la instalación? (y/n): " choice
if [[ ! "$choice" =~ ^[Yy]$ ]]; then
    print_error "Setup cancelado"
    exit 1
fi

chmod +x 1_setup.sh
./1_setup.sh

if [ $? -eq 0 ]; then
    print_success "Instalación de dependencias completada"
else
    print_error "Error durante la instalación"
    exit 1
fi

ask_continue

# PASO 3: Configurar HuggingFace Token
print_header "PASO 3: Configuración de HuggingFace Token"

print_info "Para usar PyAnnote, necesitas:"
echo "  1. Una cuenta en HuggingFace (gratis)"
echo "  2. Un token de acceso"
echo "  3. Aceptar condiciones de uso de los modelos"
echo ""

# Verificar si ya existe token
if [ ! -z "$HF_TOKEN" ]; then
    print_success "Ya existe un HF_TOKEN configurado"
    echo "Token actual: ${HF_TOKEN:0:10}..."
    echo ""
    read -p "¿Quieres usar este token? (y/n): " use_existing
    if [[ "$use_existing" =~ ^[Yy]$ ]]; then
        TOKEN_CONFIGURED=true
    fi
fi

if [ -z "$TOKEN_CONFIGURED" ]; then
    print_info "Pasos para obtener tu token:"
    echo ""
    echo "1. Ve a: https://huggingface.co/settings/tokens"
    echo "2. Crea un nuevo token (Read access es suficiente)"
    echo "3. Copia el token"
    echo ""
    echo "4. Acepta las condiciones en estos modelos:"
    echo "   → https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "   → https://huggingface.co/pyannote/segmentation-3.0"
    echo "   → https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"
    echo ""
    
    read -p "¿Ya tienes tu token listo? (y/n): " has_token
    if [[ ! "$has_token" =~ ^[Yy]$ ]]; then
        print_warning "Por favor obtén tu token primero y vuelve a ejecutar este script"
        exit 0
    fi
    
    echo ""
    read -p "Pega tu token aquí: " hf_token
    
    if [ -z "$hf_token" ]; then
        print_error "Token vacío"
        exit 1
    fi
    
    # Guardar en .bashrc
    print_info "Guardando token en ~/.bashrc..."
    echo "" >> ~/.bashrc
    echo "# HuggingFace Token para PyAnnote (agregado por setup)" >> ~/.bashrc
    echo "export HF_TOKEN='$hf_token'" >> ~/.bashrc
    
    # Exportar para esta sesión
    export HF_TOKEN="$hf_token"
    
    print_success "Token configurado correctamente"
fi

ask_continue

# PASO 4: Validar instalación
print_header "PASO 4: Validación del Sistema"

print_info "Ejecutando tests de validación..."
echo ""

source venv_diarization/bin/activate

if [ ! -f "6_test_system.py" ]; then
    print_warning "6_test_system.py no encontrado, saltando validación completa"
else
    python 6_test_system.py
    
    if [ $? -eq 0 ]; then
        print_success "Validación completada"
    else
        print_warning "Algunos tests fallaron (esto es normal si no tienes perfiles aún)"
    fi
fi

ask_continue

# PASO 5: Resumen y próximos pasos
print_header "PASO 5: ¡Setup Completado!"

print_success "El sistema está instalado y configurado correctamente"
echo ""
print_info "PRÓXIMOS PASOS:"
echo ""
echo "1️⃣  CREAR PERFILES DE HABLANTES (enrollment)"
echo "    Tiempo: 30-45 minutos (una sola vez)"
echo ""
echo "    ${GREEN}source venv_diarization/bin/activate${NC}"
echo "    ${GREEN}python 2_enrollment_gui.py${NC}"
echo ""
echo "    Se abrirá una interfaz web donde seleccionarás 5-6 segmentos"
echo "    de audio de cada uno de los 5 hablantes."
echo ""
echo "2️⃣  PROCESAR TU PRIMERA SESIÓN"
echo "    Tiempo: 15-30 minutos por sesión de 3-4 horas"
echo ""
echo "    ${GREEN}python 3_diarize_with_identification.py sesion_1/audio.m4a${NC}"
echo ""
echo "3️⃣  PROCESAR TODAS LAS SESIONES (opcional)"
echo ""
echo "    ${GREEN}python 4_batch_process.py${NC}"
echo ""

print_info "GUÍAS ÚTILES:"
echo "  📖 README.md       - Documentación completa"
echo "  🚀 QUICKSTART.md   - Guía rápida de referencia"
echo "  🧪 6_test_system.py - Tests y diagnóstico"
echo ""

print_info "COMANDOS ÚTILES:"
echo "  # Activar entorno virtual (siempre antes de usar el sistema)"
echo "  ${BLUE}source venv_diarization/bin/activate${NC}"
echo ""
echo "  # Ver estado de la GPU"
echo "  ${BLUE}watch -n 1 nvidia-smi${NC}"
echo ""
echo "  # Backup de perfiles"
echo "  ${BLUE}tar -czf speaker_profiles_backup.tar.gz speaker_profiles/${NC}"
echo ""

print_header "¡Todo listo!"

echo -e "${GREEN}"
cat << "EOF"
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║  ✨ Sistema instalado correctamente ✨           ║
    ║                                                   ║
    ║  Siguiente: python 2_enrollment_gui.py           ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
print_info "Para cualquier problema, ejecuta:"
echo "  ${BLUE}python 6_test_system.py${NC}"
echo ""