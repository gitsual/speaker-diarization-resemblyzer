#!/usr/bin/env python3
"""
diarize_speaker_mapping.py - Sistema de Mapeo 1:1 de Hablantes con Resemblyzer

DESCRIPCIÓN:
    Script principal para identificar hablantes en transcripciones. Utiliza embeddings
    de voz (Resemblyzer) para mapear cada segmento de una transcripción SRT a uno de
    los hablantes conocidos.

ALGORITMO:
    1. CARGA: Lee perfiles de hablantes (embeddings promedio de voz)
    2. PREPROCESO: Carga el audio completo en memoria una sola vez (optimización crítica)
    3. EXTRACCIÓN: Para cada segmento del SRT:
       - Extrae el audio correspondiente del array completo
       - Genera embedding(s) de voz con Resemblyzer
       - Usa multi-embedding para segmentos largos (>6s) para mayor precisión
    4. CLASIFICACIÓN: Compara cada embedding con los perfiles usando similitud coseno
    5. SUAVIZADO: Aplica ventana temporal ponderada para corregir errores puntuales
    6. EXPORTACIÓN: Genera archivo diarizado con [tiempo] HABLANTE + texto

OPTIMIZACIONES:
    - Audio cargado 1 vez (vs. 6718 veces) = 200x más rápido
    - Multi-embedding para segmentos largos = mayor precisión
    - Suavizado temporal mínimo (ventana=3) = mantiene cambios naturales
    - Procesamiento en GPU (CUDA) = velocidad óptima

PARÁMETROS CONFIGURABLES:
    MIN_SEGMENT_DURATION: Duración mínima para procesar segmento (default: 1.0s)
    SIMILARITY_THRESHOLD: Umbral de similitud para asignar hablante (default: 0.50)
    WINDOW_SIZE: Ventana para suavizado temporal (default: 3)
    USE_MULTI_EMBEDDING: Extraer múltiples embeddings en segmentos largos (default: True)

ENTRADA:
    - speaker_profiles/*_avg_embedding_resemblyzer.npy: Perfiles de hablantes
    - sesion_X/*.m4a: Audio de la sesión
    - sesion_X/*.srt: Transcripción con timestamps

SALIDA:
    - sesion_X/sesion_X_diarizada.txt: Transcripción diarizada
    - diarization.log: Log detallado del proceso

RENDIMIENTO:
    - ~200 segmentos/segundo en GPU
    - ~1.5 minutos para 2 sesiones (11,127 segmentos totales)

USO:
    python diarize_speaker_mapping.py

EJEMPLO DE SALIDA:
    [0.0s - 5.0s] HABLANTE_1
    Vale, pues en la frontera hay una fortaleza

    [5.7s - 7.2s] HABLANTE_1
    No, no, que estoy un día perdido

    [18.9s - 21.0s] HABLANTE_6
    Un contrabandista

AUTOR: Sistema de diarización con Resemblyzer optimizado
VERSIÓN: 2.0 - Mapeo 1:1 con audio precargado
"""

import numpy as np
import os
import json
import logging
import pysrt
import librosa
from pathlib import Path
from resemblyzer import preprocess_wav, VoiceEncoder
from scipy.spatial.distance import cosine
from tqdm import tqdm
from collections import Counter

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diarization.log'),  # Guardar en archivo
        logging.StreamHandler()  # Mostrar en consola
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE DIRECTORIOS
# ============================================================================
# Usa rutas relativas al script actual para portabilidad
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
PROFILES_DIR = BASE_DIR / "speaker_profiles"

# ============================================================================
# PARÁMETROS DE PROCESAMIENTO
# ============================================================================
# Duración mínima del segmento para procesarlo (segundos)
# Segmentos más cortos heredan el hablante del segmento anterior
MIN_SEGMENT_DURATION = 1.0

# Umbral de similitud coseno para asignar un hablante conocido
# Rango: 0.0 (completamente diferente) a 1.0 (idéntico)
# Si la similitud < threshold, el segmento se marca como "UNKNOWN"
SIMILARITY_THRESHOLD = 0.50

# Tamaño de ventana para suavizado temporal
# Ventana = 3 significa: segmento actual + 3 antes + 3 después
# Elementos más cercanos tienen más peso en la votación
WINDOW_SIZE = 3

# Activar multi-embedding para segmentos largos (>6 segundos)
# Extrae embeddings del inicio, medio y fin, luego promedia
# Mejora la robustez contra variaciones dentro del segmento
USE_MULTI_EMBEDDING = True


class ResemblyzerMapper:
    """
    Clase principal para mapeo de hablantes usando Resemblyzer.

    RESPONSABILIDADES:
        - Cargar y gestionar perfiles de hablantes
        - Extraer embeddings de segmentos de audio
        - Clasificar segmentos según similitud con perfiles
        - Aplicar suavizado temporal para coherencia
        - Generar archivos de salida diarizados

    ATRIBUTOS:
        profiles_dir (Path): Directorio con perfiles de hablantes
        encoder (VoiceEncoder): Modelo Resemblyzer cargado en GPU
        speaker_profiles (dict): Diccionario {nombre_hablante: embedding_promedio}

    MÉTODOS PRINCIPALES:
        load_speaker_profiles(): Carga perfiles desde disco
        extract_embedding_from_array(): Genera embedding de un segmento
        map_embedding_to_speaker(): Clasifica embedding a hablante
        temporal_smoothing(): Suaviza asignaciones temporales
        process_session(): Procesa una sesión completa
    """

    def __init__(self, profiles_dir):
        """
        Inicializa el sistema de mapeo de hablantes.

        Args:
            profiles_dir (Path): Ruta al directorio con perfiles de hablantes

        PROCESO:
            1. Carga el modelo VoiceEncoder de Resemblyzer en GPU
            2. Lee todos los perfiles de hablantes disponibles
            3. Muestra información de configuración

        NOTAS:
            - Requiere GPU con CUDA para rendimiento óptimo
            - Sin GPU, usar device="cpu" pero será mucho más lento
        """
        self.profiles_dir = Path(profiles_dir)

        # Cargar el modelo de extracción de embeddings
        logger.info("Cargando Resemblyzer encoder...")
        self.encoder = VoiceEncoder(device="cuda")  # Usar GPU
        logger.info("✅ Encoder cargado")

        # Cargar todos los perfiles de hablantes conocidos
        self.speaker_profiles = self.load_speaker_profiles()
        logger.info(f"✅ Cargados {len(self.speaker_profiles)} perfiles de hablantes:")
        for name in self.speaker_profiles:
            logger.info(f"   - {name}")

    def load_speaker_profiles(self):
        """
        Carga los perfiles de hablantes desde disco.

        Returns:
            dict: Diccionario {speaker_name: embedding_array}

        PROCESO:
            1. Busca archivos *_profile.json en profiles_dir
            2. Para cada perfil, lee el nombre del hablante
            3. Carga el embedding promedio de Resemblyzer (.npy)
            4. Almacena en diccionario

        FORMATO DE ARCHIVOS:
            - *_profile.json: Metadatos del perfil
            - *_avg_embedding_resemblyzer.npy: Embedding promedio (NumPy array)

        NOTAS:
            - Solo carga perfiles con embeddings de Resemblyzer
            - Ignora perfiles sin archivo .npy
        """
        profiles = {}

        # Buscar todos los archivos de perfil
        for profile_file in sorted(self.profiles_dir.glob("*_profile.json")):
            # Leer metadatos del perfil
            with open(profile_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)

            speaker_name = profile_data['speaker_id']

            # Cargar embedding promedio de Resemblyzer
            avg_emb_path = self.profiles_dir / f"{speaker_name}_avg_embedding_resemblyzer.npy"
            if avg_emb_path.exists():
                profiles[speaker_name] = np.load(avg_emb_path)

        return profiles

    def extract_embedding_from_array(self, wav_full, sr, start_sec, end_sec):
        """
        Extrae embedding de voz de un segmento desde el audio completo precargado.

        Args:
            wav_full (np.array): Audio completo precargado en memoria
            sr (int): Sample rate del audio (debe ser 16000 Hz)
            start_sec (float): Tiempo de inicio del segmento en segundos
            end_sec (float): Tiempo de fin del segmento en segundos

        Returns:
            np.array or None: Embedding de voz (256 dimensiones) o None si falla

        ALGORITMO:
            1. Calcula índices de muestra desde timestamps
            2. Extrae el segmento del array completo (slicing)
            3. Preprocesa el audio (normalización, filtros)
            4. Si USE_MULTI_EMBEDDING y duración >= 6s:
               - Divide en 3 partes solapadas
               - Extrae embedding de cada parte
               - Promedia los embeddings
            5. Si no, extrae embedding simple del segmento completo

        MULTI-EMBEDDING:
            Para segmentos largos (≥6s), extrae embeddings de:
            - Primera 2/3 partes (inicio + medio)
            - Mitad central
            - Última 2/3 partes (medio + final)
            Esto captura variaciones de voz a lo largo del segmento.

        OPTIMIZACIÓN:
            Al usar el audio precargado, evita cargar el archivo M4A 6718 veces.
            Esto reduce el tiempo de 2+ horas a ~1.5 minutos.

        EXCEPCIONES:
            - Si el segmento < 1 segundo, retorna None
            - Si hay error en extracción, registra en debug y retorna None
        """
        try:
            # Calcular duración del segmento
            duration = end_sec - start_sec

            # Convertir timestamps a índices de muestra
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            # Extraer segmento del audio completo (operación rápida en NumPy)
            wav = wav_full[start_sample:end_sample]

            # Verificar duración mínima
            if len(wav) < sr:  # Menos de 1 segundo
                return None

            # Preprocesar audio (normalización, filtrado de ruido)
            wav = preprocess_wav(wav)

            # ESTRATEGIA MULTI-EMBEDDING para segmentos largos
            if USE_MULTI_EMBEDDING and duration >= 6.0:
                embeddings = []

                # Dividir en tres partes
                third = len(wav) // 3

                # Verificar que cada parte tenga al menos 1 segundo
                if third >= 16000:  # 16000 muestras = 1 segundo a 16kHz
                    # Extraer embedding de la primera 2/3 partes (inicio + medio)
                    emb1 = self.encoder.embed_utterance(wav[:third*2])
                    embeddings.append(emb1)

                    # Extraer embedding de la parte central (con solape)
                    start_middle = third // 2
                    end_middle = start_middle + third * 2
                    if end_middle <= len(wav):
                        emb2 = self.encoder.embed_utterance(wav[start_middle:end_middle])
                        embeddings.append(emb2)

                    # Extraer embedding de la última 2/3 partes (medio + final)
                    emb3 = self.encoder.embed_utterance(wav[third:])
                    embeddings.append(emb3)

                    # Promediar todos los embeddings para robustez
                    embedding = np.mean(embeddings, axis=0)
                else:
                    # Si las partes son muy cortas, usar el segmento completo
                    embedding = self.encoder.embed_utterance(wav)
            else:
                # ESTRATEGIA SIMPLE: extraer embedding del segmento completo
                embedding = self.encoder.embed_utterance(wav)

            return embedding

        except Exception as e:
            # Registrar error pero no detener el procesamiento
            logger.debug(f"Error extrayendo embedding ({start_sec}-{end_sec}s): {e}")
            return None

    def map_embedding_to_speaker(self, embedding):
        """
        Mapea un embedding de voz al hablante más similar.

        Args:
            embedding (np.array): Embedding de voz a clasificar (256 dimensiones)

        Returns:
            tuple: (nombre_hablante, similitud)
                - nombre_hablante (str): Nombre del hablante o "UNKNOWN"
                - similitud (float): Similitud coseno (0.0 a 1.0)

        ALGORITMO:
            1. Compara embedding con cada perfil de hablante
            2. Usa similitud coseno: 1 - distancia_coseno
            3. Selecciona el hablante con mayor similitud
            4. Si similitud >= SIMILARITY_THRESHOLD: asigna hablante
            5. Si similitud < threshold: marca como "UNKNOWN"

        SIMILITUD COSENO:
            - Mide el ángulo entre dos vectores
            - 1.0 = vectores idénticos (mismo hablante)
            - 0.5 = vectores ortogonales (hablantes diferentes)
            - 0.0 = vectores opuestos (muy diferentes)

        THRESHOLD:
            - 0.50 (default): Balance entre precisión y cobertura
            - Valores más altos: Menos falsos positivos, más "UNKNOWN"
            - Valores más bajos: Más asignaciones, más falsos positivos

        CASOS ESPECIALES:
            - Si embedding es None: retorna (None, 0.0)
            - Si no hay perfiles cargados: retorna ("UNKNOWN", 0.0)
        """
        # Verificar que hay embedding válido
        if embedding is None:
            return None, 0.0

        best_match = None
        best_similarity = -1

        # Comparar con cada perfil de hablante conocido
        for speaker_name, profile_emb in self.speaker_profiles.items():
            # Calcular similitud coseno
            # scipy.spatial.distance.cosine retorna la distancia (0 = idénticos)
            # Convertimos a similitud: similitud = 1 - distancia
            similarity = 1 - cosine(embedding, profile_emb)

            # Actualizar mejor coincidencia si esta es mejor
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_name

        # Verificar si la mejor similitud supera el threshold
        if best_similarity >= SIMILARITY_THRESHOLD:
            return best_match, best_similarity
        else:
            # Similitud insuficiente: marcar como desconocido
            return "UNKNOWN", best_similarity

    def temporal_smoothing(self, speaker_assignments, window_size=WINDOW_SIZE):
        """
        Aplica suavizado temporal con ponderación por proximidad.

        Args:
            speaker_assignments (list): Lista de asignaciones de hablantes
            window_size (int): Tamaño de ventana a cada lado (default: WINDOW_SIZE)

        Returns:
            list: Lista suavizada de asignaciones

        PROPÓSITO:
            Corregir errores puntuales de clasificación manteniendo cambios naturales.
            Si un segmento está rodeado de segmentos del mismo hablante, probablemente
            sea el mismo hablante (error de clasificación).

        ALGORITMO:
            1. Para cada segmento i:
               - Define ventana: [i-window_size, i+window_size]
               - Cuenta votos de cada hablante en la ventana
               - Pondera votos por distancia: más cercano = más peso
               - Asigna el hablante con más votos ponderados

        PONDERACIÓN:
            - Peso = window_size - distancia + 1
            - Segmento actual (distancia=0): peso = window_size + 1
            - Segmento a distancia 1: peso = window_size
            - Segmento a distancia window_size: peso = 1

        EJEMPLO (window_size=3):
            Entrada:  [A, A, A, B, A, A, A]
            Ventana en i=3: [A, A, A, B, A, A, A]
            Votos ponderados:
                A: (2*1) + (3*1) + (4*1) + (4*1) + (3*1) + (2*1) = 18
                B: 4*1 = 4
            Resultado en i=3: A (corrige error puntual)

        VENTAJAS:
            - Preserva cambios genuinos de hablante
            - Suaviza errores puntuales
            - Ventana pequeña (3) = mínima interferencia
        """
        smoothed = []

        for i in range(len(speaker_assignments)):
            # Definir límites de la ventana
            start = max(0, i - window_size)
            end = min(len(speaker_assignments), i + window_size + 1)
            window = speaker_assignments[start:end]

            # Contador de votos ponderados por hablante
            weighted_votes = Counter()

            # Procesar cada elemento en la ventana
            for idx, speaker in enumerate(window):
                # Calcular distancia al segmento actual
                # min(...) maneja el caso cuando i está cerca del borde
                distance = abs(idx - min(i - start, len(window) - 1))

                # Calcular peso: más cercano = más peso
                weight = window_size - distance + 1

                # Agregar voto ponderado
                weighted_votes[speaker] += weight

            # Seleccionar hablante con más votos ponderados
            most_common = weighted_votes.most_common(1)[0][0]
            smoothed.append(most_common)

        return smoothed

    def process_session(self, session_dir):
        """
        Procesa una sesión completa de audio + transcripción.

        Args:
            session_dir (Path): Directorio de la sesión (ej: sesion_1/)

        Returns:
            bool: True si procesamiento exitoso, False si hay errores

        PROCESO COMPLETO:
            1. VALIDACIÓN:
               - Busca archivos .m4a y .srt en el directorio
               - Verifica que existan ambos archivos

            2. CARGA DE DATOS:
               - Lee transcripción SRT con timestamps
               - Carga audio completo en memoria (¡UNA SOLA VEZ!)

            3. EXTRACCIÓN DE EMBEDDINGS:
               - Para cada segmento del SRT:
                 * Extrae audio desde el array precargado
                 * Genera embedding de voz
                 * Clasifica a hablante más similar
               - Usa tqdm para barra de progreso

            4. SUAVIZADO TEMPORAL:
               - Aplica ventana ponderada para coherencia

            5. EXPORTACIÓN:
               - Genera archivo *_diarizada.txt con formato:
                 [inicio - fin] HABLANTE
                 texto del segmento

            6. ESTADÍSTICAS:
               - Muestra distribución de hablantes
               - Calcula porcentajes por hablante

        ARCHIVOS ENTRADA:
            - sesion_X/*.m4a: Audio de la sesión
            - sesion_X/*.srt: Transcripción con timestamps

        ARCHIVOS SALIDA:
            - sesion_X/sesion_X_diarizada.txt: Transcripción diarizada

        OPTIMIZACIÓN CRÍTICA:
            Cargar el audio UNA VEZ al inicio es la clave del rendimiento.
            Antes: librosa.load() por cada segmento = 6718 cargas = 2+ horas
            Ahora: librosa.load() una vez + slicing = ~1.5 minutos

        MANEJO DE SEGMENTOS CORTOS:
            - Si segmento < MIN_SEGMENT_DURATION:
              * Hereda hablante del segmento anterior
              * Evita embeddings de baja calidad

        EXCEPCIONES:
            - Registra errores pero continúa procesamiento
            - Retorna False si faltan archivos esenciales
        """
        session_name = session_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Procesando: {session_name}")
        logger.info(f"{'='*60}")

        # ====================================================================
        # PASO 1: VALIDAR Y LOCALIZAR ARCHIVOS
        # ====================================================================
        audio_files = list(session_dir.glob("*.m4a"))
        srt_files = list(session_dir.glob("*.srt"))

        if not audio_files or not srt_files:
            logger.error(f"❌ Faltan archivos en {session_dir}")
            return False

        audio_path = audio_files[0]
        srt_path = srt_files[0]

        logger.info(f"🎵 Audio: {audio_path.name}")
        logger.info(f"📄 Subtítulos: {srt_path.name}")

        # ====================================================================
        # PASO 2: CARGAR TRANSCRIPCIÓN SRT
        # ====================================================================
        subs = pysrt.open(srt_path)
        logger.info(f"✅ {len(subs)} segmentos en SRT")

        # ====================================================================
        # PASO 3: CARGAR AUDIO COMPLETO (OPTIMIZACIÓN CRÍTICA)
        # ====================================================================
        logger.info("Cargando audio completo...")
        wav_full, sr = librosa.load(str(audio_path), sr=16000)
        logger.info(f"✅ Audio cargado: {len(wav_full)/sr:.1f}s")

        # ====================================================================
        # PASO 4: PROCESAR TODOS LOS SEGMENTOS (MAPEO 1:1)
        # ====================================================================
        logger.info("Procesando todos los segmentos...")
        all_speakers = []

        # Barra de progreso con tqdm
        for sub in tqdm(subs, desc="  Extrayendo embeddings"):
            # Convertir timestamps a segundos
            start_sec = sub.start.ordinal / 1000.0
            end_sec = sub.end.ordinal / 1000.0
            duration = end_sec - start_sec

            # Procesar solo segmentos con duración suficiente
            if duration >= MIN_SEGMENT_DURATION:
                # Extraer embedding desde el audio precargado
                embedding = self.extract_embedding_from_array(wav_full, sr, start_sec, end_sec)

                # Clasificar a hablante
                speaker, similarity = self.map_embedding_to_speaker(embedding)

                # Agregar resultado (None se convierte en "UNKNOWN")
                all_speakers.append(speaker if speaker else "UNKNOWN")
            else:
                # Segmento muy corto: heredar hablante del anterior
                if all_speakers:
                    all_speakers.append(all_speakers[-1])
                else:
                    # Primer segmento y es muy corto
                    all_speakers.append("UNKNOWN")

        logger.info(f"✅ {len(all_speakers)} segmentos procesados")

        # ====================================================================
        # PASO 5: SUAVIZADO TEMPORAL
        # ====================================================================
        logger.info("Suavizado temporal...")
        smoothed_speakers = self.temporal_smoothing(all_speakers)

        # ====================================================================
        # PASO 6: GUARDAR RESULTADO DIARIZADO
        # ====================================================================
        output_file = session_dir / f"{session_name}_diarizada.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subs):
                # Obtener timestamps y hablante
                start_sec = sub.start.ordinal / 1000.0
                end_sec = sub.end.ordinal / 1000.0
                speaker = smoothed_speakers[i]
                text = sub.text.strip()

                # Formato: [inicio - fin] HABLANTE\ntexto\n\n
                line = f"[{start_sec:.1f}s - {end_sec:.1f}s] {speaker}\n{text}\n\n"
                f.write(line)

        logger.info(f"💾 Guardado: {output_file}")

        # ====================================================================
        # PASO 7: MOSTRAR ESTADÍSTICAS
        # ====================================================================
        speaker_counts = Counter(smoothed_speakers)
        logger.info(f"\n📊 RESUMEN:")
        logger.info(f"   Total segmentos: {len(subs)}")
        logger.info(f"   Distribución:")
        for speaker, count in speaker_counts.most_common():
            percentage = (count / len(subs)) * 100
            logger.info(f"      - {speaker}: {count} ({percentage:.1f}%)")

        return True


def main():
    """
    Función principal que ejecuta el sistema de diarización completo.

    PROCESO:
        1. Muestra configuración y parámetros
        2. Inicializa el sistema ResemblyzerMapper
        3. Busca todas las sesiones (directorios sesion_*)
        4. Procesa cada sesión secuencialmente
        5. Maneja errores por sesión sin detener el flujo

    CONFIGURACIÓN MOSTRADA:
        - Modo de procesamiento (1:1 completo)
        - Duración mínima de segmento
        - Threshold de similitud
        - Tamaño de ventana de suavizado
        - Estado de multi-embedding

    DIRECTORIOS PROCESADOS:
        - Busca en BASE_DIR todos los directorios que empiecen con "sesion_"
        - Los procesa en orden alfabético

    MANEJO DE ERRORES:
        - Captura excepciones por sesión
        - Registra error con stack trace
        - Continúa con la siguiente sesión

    SALIDA:
        - Logs en consola y archivo diarization.log
        - Archivos *_diarizada.txt por sesión
        - Mensaje de completado al final
    """

    # Mostrar encabezado y configuración
    logger.info("\n" + "="*60)
    logger.info("🎭 MAPEO 1:1 CON RESEMBLYZER")
    logger.info("="*60)
    logger.info(f"Configuración:")
    logger.info(f"  - Modo: Procesamiento completo (todos los segmentos)")
    logger.info(f"  - Duración mínima: {MIN_SEGMENT_DURATION}s")
    logger.info(f"  - Threshold similitud: {SIMILARITY_THRESHOLD}")
    logger.info(f"  - Ventana suavizado: {WINDOW_SIZE}")
    logger.info(f"  - Multi-embedding: {USE_MULTI_EMBEDDING}")
    logger.info("="*60 + "\n")

    # Inicializar sistema de mapeo
    system = ResemblyzerMapper(PROFILES_DIR)

    # Buscar todas las sesiones disponibles
    sessions = sorted([d for d in BASE_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sesion_")])

    logger.info(f"Encontradas {len(sessions)} sesiones:")
    for session in sessions:
        logger.info(f"  - {session.name}")

    # Procesar cada sesión
    for session_dir in sessions:
        try:
            system.process_session(session_dir)
        except Exception as e:
            # Registrar error pero continuar con siguiente sesión
            logger.error(f"❌ Error: {e}", exc_info=True)

    logger.info("\n✅ Completado")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    """
    Ejecuta el script solo cuando se invoca directamente.
    No se ejecuta si el módulo es importado.
    """
    main()
