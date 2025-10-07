#!/usr/bin/env python3
"""
convert_profiles.py - Convertidor de Perfiles de Hablantes a Embeddings Resemblyzer

DESCRIPCIÓN:
    Este script convierte los perfiles de hablantes creados con PyAnnote (enrollment_gui.py)
    a embeddings de Resemblyzer. Es necesario ejecutarlo una sola vez después de crear
    los perfiles iniciales.

PROPÓSITO:
    Resemblyzer es más rápido y estable que PyAnnote para el mapeo 1:1 de segmentos.
    Este script recalcula los embeddings de voz usando el modelo Resemblyzer para
    cada segmento de audio almacenado en los perfiles.

FLUJO DE TRABAJO:
    1. Carga el modelo VoiceEncoder de Resemblyzer en GPU (CUDA)
    2. Lee todos los archivos *_profile.json en speaker_profiles/
    3. Para cada hablante:
       - Carga los segmentos de audio (.wav)
       - Extrae embeddings usando Resemblyzer
       - Calcula el embedding promedio
       - Guarda el embedding promedio como *_avg_embedding_resemblyzer.npy

REQUISITOS:
    - GPU con CUDA (el modelo se carga en 'cuda')
    - Perfiles de hablantes creados previamente con enrollment_gui.py
    - Archivos de audio .wav en speaker_profiles/

USO:
    python convert_profiles.py

SALIDA:
    - Archivos *_avg_embedding_resemblyzer.npy en speaker_profiles/
    - Logs de progreso en consola

AUTOR: Sistema de diarización con Resemblyzer
"""

import numpy as np
import json
import logging
from pathlib import Path
from resemblyzer import preprocess_wav, VoiceEncoder
import librosa

# Configuración de logging para mostrar mensajes informativos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directorio donde se encuentran los perfiles de hablantes
# Usa ruta relativa al script actual para portabilidad
SCRIPT_DIR = Path(__file__).resolve().parent
PROFILES_DIR = SCRIPT_DIR / "speaker_profiles"


def main():
    """
    Función principal que coordina la conversión de perfiles.

    PROCESO:
        1. Inicializa el encoder de Resemblyzer en GPU
        2. Busca todos los archivos de perfil (*_profile.json)
        3. Para cada perfil:
           - Lee los metadatos del hablante
           - Procesa cada segmento de audio
           - Calcula el embedding promedio
           - Guarda el resultado

    ARCHIVOS ENTRADA:
        - speaker_profiles/*_profile.json: Metadatos de segmentos por hablante
        - speaker_profiles/*_segment_*.wav: Audio de cada segmento

    ARCHIVOS SALIDA:
        - speaker_profiles/*_avg_embedding_resemblyzer.npy: Embedding promedio

    EXCEPCIONES:
        - No maneja excepciones explícitamente, dejará fallar si hay errores
    """

    # Cargar el modelo VoiceEncoder de Resemblyzer
    # Este modelo convierte voz a vectores de características (embeddings)
    logger.info("Cargando Resemblyzer encoder...")
    encoder = VoiceEncoder(device="cuda")  # Usa GPU para mayor velocidad
    logger.info("✅ Encoder cargado")

    # Buscar todos los archivos de perfil en orden alfabético
    for profile_file in sorted(PROFILES_DIR.glob("*_profile.json")):

        # Leer el archivo JSON con los metadatos del perfil
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)

        # Obtener el nombre del hablante (ej: "HABLANTE_1")
        speaker_name = profile_data['speaker_id']
        logger.info(f"\nProcesando: {speaker_name}")

        # Lista para almacenar todos los embeddings de este hablante
        embeddings = []

        # Procesar cada segmento de audio del hablante
        for segment in profile_data['segments']:
            # Obtener la ruta del archivo de audio
            # Tomamos solo el nombre del archivo (última parte de la ruta)
            audio_path = PROFILES_DIR / segment['audio_path'].split('/')[-1]

            # Verificar que el archivo existe
            if audio_path.exists():
                # Cargar audio con librosa a 16kHz (frecuencia requerida por Resemblyzer)
                wav, sr = librosa.load(str(audio_path), sr=16000)

                # Preprocesar el audio (normalización, filtrado)
                wav = preprocess_wav(wav)

                # Extraer el embedding de voz (vector de características)
                # Este vector representa las características únicas de la voz
                embedding = encoder.embed_utterance(wav)
                embeddings.append(embedding)

                logger.info(f"   ✅ {audio_path.name}")

        # Si se procesaron embeddings, calcular y guardar el promedio
        if embeddings:
            # Calcular el embedding promedio de todos los segmentos
            # Esto crea un "perfil de voz" representativo del hablante
            avg_embedding = np.mean(embeddings, axis=0)

            # Guardar el embedding promedio como archivo .npy (formato NumPy)
            output_path = PROFILES_DIR / f"{speaker_name}_avg_embedding_resemblyzer.npy"
            np.save(output_path, avg_embedding)
            logger.info(f"💾 Guardado: {output_path.name}")

    logger.info("\n✅ Conversión completada")


if __name__ == "__main__":
    # Ejecutar la función principal solo si este script se ejecuta directamente
    # (no si se importa como módulo)
    main()
