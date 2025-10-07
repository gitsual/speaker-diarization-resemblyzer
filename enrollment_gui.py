#!/usr/bin/env python3
"""
GUI para selección manual de segmentos de enrollment
Fase 1: Permite escuchar y seleccionar clips de cada hablante
"""

import gradio as gr
import numpy as np
# Compatibilidad con NumPy 2.0: algunos paquetes (p.ej., pyannote) usan np.NaN
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import soundfile as sf
import json
import os
from pathlib import Path
from datetime import datetime
import torch
import librosa
from pydub import AudioSegment
from pyannote.audio import Model, Inference
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuración
SPEAKERS = [
    "HABLANTE_1",
    "HABLANTE_2",
    "HABLANTE_3",
    "HABLANTE_4",
    "HABLANTE_5",
    "HABLANTE_6",
]
PROFILES_DIR = Path("speaker_profiles")
TEMP_DIR = Path("temp_audio_segments")
SAMPLE_RATE = 16000

# Crear directorios
PROFILES_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Cargar modelo de embeddings
print("Cargando modelo de embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

embedding_model = Model.from_pretrained(
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    use_auth_token=os.getenv("HF_TOKEN")
)
inference = Inference(embedding_model, window="whole", device=device)

class EnrollmentManager:
    def __init__(self):
        self.current_audio = None
        self.current_sr = None
        self.audio_path = None
        self.enrollments = {speaker: [] for speaker in SPEAKERS}
        self.load_existing_profiles()
    
    @staticmethod
    def hms_to_seconds(hours, minutes, seconds):
        try:
            h = int(hours or 0)
            m = int(minutes or 0)
            s = float(seconds or 0)
            if h < 0 or m < 0 or s < 0:
                return None
            return h * 3600 + m * 60 + s
        except Exception:
            return None
    
    def load_existing_profiles(self):
        """Cargar perfiles existentes si existen"""
        for speaker in SPEAKERS:
            profile_path = PROFILES_DIR / f"{speaker}_profile.json"
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    self.enrollments[speaker] = data.get('segments', [])
    
    def load_audio_file(self, audio_path):
        """Cargar archivo de audio (.m4a, .wav, etc)"""
        try:
            if not os.path.exists(audio_path):
                return None, "Archivo no encontrado"
            
            # Convertir a WAV temporal si es necesario
            audio_path = Path(audio_path)
            if audio_path.suffix.lower() == '.m4a':
                print(f"Convirtiendo {audio_path.name} a WAV...")
                audio = AudioSegment.from_file(str(audio_path), format="m4a")
                temp_wav = TEMP_DIR / f"{audio_path.stem}_temp.wav"
                audio.export(str(temp_wav), format="wav")
                audio_path = temp_wav
            
            # Cargar audio
            self.current_audio, self.current_sr = librosa.load(
                str(audio_path), 
                sr=SAMPLE_RATE,
                mono=True
            )
            self.audio_path = audio_path
            
            duration = len(self.current_audio) / self.current_sr
            
            return (
                self.current_sr, 
                (self.current_audio * 32767).astype(np.int16)
            ), f"✅ Audio cargado: {duration:.1f} segundos"
            
        except Exception as e:
            return None, f"❌ Error cargando audio: {str(e)}"
    
    def extract_segment(self, start_time, end_time):
        """Extraer segmento de audio entre start y end (en segundos)"""
        if self.current_audio is None:
            return None, "⚠️ Primero carga un archivo de audio"
        
        try:
            start_time = float(start_time)
            end_time = float(end_time)
            
            if start_time >= end_time:
                return None, "❌ El tiempo de inicio debe ser menor que el final"
            
            if end_time - start_time < 5:
                return None, "❌ El segmento debe tener al menos 5 segundos"
            
            if end_time - start_time > 30:
                return None, "⚠️ Segmento muy largo (>30s). Recomendado: 5-15s"
            
            # Extraer segmento
            start_sample = int(start_time * self.current_sr)
            end_sample = int(end_time * self.current_sr)
            
            segment_audio = self.current_audio[start_sample:end_sample]
            
            # Verificar que tenga contenido
            if np.max(np.abs(segment_audio)) < 0.01:
                return None, "⚠️ El segmento parece estar en silencio"
            
            return (
                self.current_sr,
                (segment_audio * 32767).astype(np.int16)
            ), f"✅ Segmento extraído: {end_time - start_time:.1f}s"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def save_enrollment_segment(self, speaker, start_time, end_time, notes=""):
        """Guardar segmento de enrollment para un hablante"""
        if self.current_audio is None:
            return "⚠️ Primero carga un archivo de audio"
        
        try:
            start_time = float(start_time)
            end_time = float(end_time)
            
            # Extraer y guardar audio
            start_sample = int(start_time * self.current_sr)
            end_sample = int(end_time * self.current_sr)
            segment_audio = self.current_audio[start_sample:end_sample]
            
            # Guardar WAV temporal
            existing_ids = [seg.get('segment_id', 0) for seg in self.enrollments[speaker]]
            segment_id = (max(existing_ids) + 1) if existing_ids else 1
            segment_path = PROFILES_DIR / f"{speaker}_segment_{segment_id}.wav"
            sf.write(segment_path, segment_audio, self.current_sr)
            
            # Extraer embedding
            print(f"Extrayendo embedding para {speaker} segmento {segment_id}...")
            embedding = inference({"audio": str(segment_path)})
            embedding_path = PROFILES_DIR / f"{speaker}_segment_{segment_id}.npy"
            np.save(embedding_path, embedding)
            
            # Metadata
            segment_info = {
                'segment_id': segment_id,
                'audio_file': str(self.audio_path.name),
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'notes': notes,
                'timestamp': datetime.now().isoformat(),
                'audio_path': str(segment_path),
                'embedding_path': str(embedding_path)
            }
            
            self.enrollments[speaker].append(segment_info)
            
            # Guardar perfil actualizado
            self.save_speaker_profile(speaker)
            
            count = len(self.enrollments[speaker])
            return f"✅ Segmento {segment_id} guardado para {speaker} ({count}/5-6 segmentos)"
            
        except Exception as e:
            return f"❌ Error guardando segmento: {str(e)}"

    def find_segment_index(self, speaker, segment_id):
        try:
            sid = int(segment_id)
        except Exception:
            return -1, None
        for idx, seg in enumerate(self.enrollments.get(speaker, [])):
            if int(seg.get('segment_id', -1)) == sid:
                return idx, seg
        return -1, None

    def delete_segment(self, speaker, segment_id):
        idx, seg = self.find_segment_index(speaker, segment_id)
        if seg is None:
            return f"❌ {speaker}: segmento {segment_id} no encontrado"
        # Borrar archivos asociados
        for key in ['audio_path', 'embedding_path']:
            p = seg.get(key)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        # Eliminar del registro y guardar perfil
        del self.enrollments[speaker][idx]
        self.save_speaker_profile(speaker)
        return f"🗑️ Segmento {segment_id} de {speaker} eliminado"

    def load_segment_audio(self, speaker, segment_id):
        idx, seg = self.find_segment_index(speaker, segment_id)
        if seg is None:
            return None, "❌ Segmento no encontrado"
        path = seg.get('audio_path')
        if not path or not os.path.exists(path):
            return None, "❌ Archivo de audio no disponible"
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            return (sr, (y * 32767).astype(np.int16)), f"▶️ Segmento {segment_id} listo para reproducción"
        except Exception as e:
            return None, f"❌ Error cargando audio: {str(e)}"
    
    def save_speaker_profile(self, speaker):
        """Guardar perfil completo del hablante"""
        profile_path = PROFILES_DIR / f"{speaker}_profile.json"
        
        # Calcular embedding promedio si hay suficientes segmentos
        embeddings = []
        for seg in self.enrollments[speaker]:
            emb_path = Path(seg['embedding_path'])
            if emb_path.exists():
                embeddings.append(np.load(emb_path))
        
        avg_embedding = None
        if len(embeddings) >= 3:
            avg_embedding_path = PROFILES_DIR / f"{speaker}_avg_embedding.npy"
            avg_emb = np.mean(embeddings, axis=0)
            np.save(avg_embedding_path, avg_emb)
            avg_embedding = str(avg_embedding_path)
        
        profile = {
            'speaker_id': speaker,
            'num_segments': len(self.enrollments[speaker]),
            'segments': self.enrollments[speaker],
            'avg_embedding_path': avg_embedding,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def load_embeddings_for_speaker(self, speaker):
        """Cargar embeddings y metadatos de un hablante"""
        embeddings = []
        rows = []
        for seg in self.enrollments.get(speaker, []):
            emb_path = Path(seg['embedding_path'])
            if emb_path.exists():
                try:
                    emb = np.load(emb_path)
                    embeddings.append(emb)
                    rows.append([
                        seg['segment_id'],
                        seg['start_time'],
                        seg['end_time'],
                        seg['duration'],
                        seg.get('notes', ''),
                        str(emb_path)
                    ])
                except Exception:
                    continue
        return embeddings, rows

    @staticmethod
    def pca_2d(array_list):
        """Proyección PCA 2D simple (sin sklearn)"""
        if not array_list:
            return None
        X = np.vstack([a.reshape(1, -1) if a.ndim > 1 else a.reshape(1, -1) for a in array_list])
        X = X.astype(np.float64)
        X -= X.mean(axis=0, keepdims=True)
        # SVD
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            components = Vt[:2]
            X2 = X @ components.T
            return X2
        except Exception:
            return None
    
    def get_speaker_status(self):
        """Obtener estado de enrollment de todos los hablantes"""
        status = []
        for speaker in SPEAKERS:
            count = len(self.enrollments[speaker])
            status_emoji = "✅" if count >= 5 else "⚠️" if count >= 3 else "❌"
            status.append(f"{status_emoji} {speaker}: {count} segmentos")
        return "\n".join(status)

# Instancia global
manager = EnrollmentManager()

# Funciones para Gradio
def load_audio(audio_file):
    audio_data, msg = manager.load_audio_file(audio_file)
    status = manager.get_speaker_status()
    return audio_data, msg, status

def preview_segment(start_h, start_m, start_s, end_h, end_m, end_s):
    start = manager.hms_to_seconds(start_h, start_m, start_s)
    end = manager.hms_to_seconds(end_h, end_m, end_s)
    if start is None or end is None:
        return None, "❌ Formato de tiempo inválido"
    audio_data, msg = manager.extract_segment(start, end)
    return audio_data, msg

def save_segment(speaker, start_h, start_m, start_s, end_h, end_m, end_s, notes):
    start = manager.hms_to_seconds(start_h, start_m, start_s)
    end = manager.hms_to_seconds(end_h, end_m, end_s)
    if start is None or end is None:
        return "❌ Formato de tiempo inválido", manager.get_speaker_status()
    msg = manager.save_enrollment_segment(speaker, start, end, notes)
    status = manager.get_speaker_status()
    return msg, status

def view_embeddings(speaker):
    embeddings, rows = manager.load_embeddings_for_speaker(speaker)
    if not embeddings:
        return None, f"❌ {speaker}: no hay embeddings", []
    proj = manager.pca_2d(embeddings)
    fig = None
    if proj is not None and proj.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(proj[:, 0], proj[:, 1], c='tab:blue', label='segments')
        for i, row in enumerate(rows):
            seg_id = row[0]
            ax.annotate(str(seg_id), (proj[i, 0], proj[i, 1]), fontsize=8, alpha=0.7)
        ax.set_title(f"Embeddings PCA 2D - {speaker}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
    info = f"✅ {speaker}: {len(embeddings)} embeddings cargados"
    return fig, info, rows

# Crear interfaz Gradio
with gr.Blocks(title="Sistema de Enrollment de Hablantes") as demo:
    gr.Markdown("# 🎙️ Sistema de Enrollment de Hablantes")
    gr.Markdown("""
    **Objetivo:** Seleccionar 5-6 segmentos de audio de cada hablante (5-15 segundos cada uno).
    
    **Instrucciones:**
    1. Carga un archivo de audio (.m4a, .wav)
    2. Escucha y localiza segmentos donde habla claramente UN hablante específico
    3. Anota los tiempos (inicio y fin en segundos)
    4. Selecciona el hablante y guarda el segmento
    5. Repite hasta tener 5-6 segmentos por hablante
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Carga de audio
            gr.Markdown("### 1️⃣ Cargar Audio")
            audio_input = gr.File(label="Archivo de audio (.m4a, .wav)", file_types=[".m4a", ".wav"])
            audio_player = gr.Audio(label="Audio completo", interactive=False)
            load_msg = gr.Textbox(label="Estado", interactive=False)
            
            # Selección de segmento
            gr.Markdown("### 2️⃣ Seleccionar Segmento (H:M:S)")
            with gr.Row():
                start_h = gr.Number(label="Inicio - Horas", value=0, precision=0)
                start_m = gr.Number(label="Inicio - Min", value=0, precision=0)
                start_s = gr.Number(label="Inicio - Seg", value=0, precision=1)
            with gr.Row():
                end_h = gr.Number(label="Fin - Horas", value=0, precision=0)
                end_m = gr.Number(label="Fin - Min", value=0, precision=0)
                end_s = gr.Number(label="Fin - Seg", value=10, precision=1)
            
            preview_btn = gr.Button("👂 Previsualizar Segmento", variant="secondary")
            segment_player = gr.Audio(label="Previsualización del segmento")
            preview_msg = gr.Textbox(label="Estado", interactive=False)
            
            # Guardar enrollment
            gr.Markdown("### 3️⃣ Guardar Segmento de Enrollment")
            speaker_dropdown = gr.Dropdown(
                choices=SPEAKERS,
                label="Hablante",
                value=SPEAKERS[0]
            )
            notes_input = gr.Textbox(
                label="Notas (opcional)",
                placeholder="Ej: voz tranquila, al inicio de sesión, etc."
            )
            save_btn = gr.Button("💾 Guardar Segmento de Enrollment", variant="primary")
            save_msg = gr.Textbox(label="Resultado", interactive=False)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Estado del Enrollment")
            status_display = gr.Textbox(
                label="Progreso por Hablante",
                value=manager.get_speaker_status(),
                interactive=False,
                lines=10
            )
            
            gr.Markdown("""
            **Leyenda:**
            - ✅ 5+ segmentos (óptimo)
            - ⚠️ 3-4 segmentos (mínimo aceptable)
            - ❌ <3 segmentos (insuficiente)
            
            **Consejos:**
            - Busca momentos donde el hablante habla SOLO
            - Varía los contextos (formal, casual, emocional)
            - Evita segmentos con ruido excesivo
            - 5-10 segundos por segmento es ideal
            """)

            gr.Markdown("### 🔎 Embeddings por Hablante")
            emb_speaker = gr.Dropdown(choices=SPEAKERS, label="Hablante", value=SPEAKERS[0])
            emb_btn = gr.Button("📈 Ver embeddings")
            emb_plot = gr.Plot(label="Proyección PCA 2D")
            emb_info = gr.Textbox(label="Info", interactive=False)
            emb_table = gr.Dataframe(
                headers=["segment_id","start","end","dur","notes","embedding_path"],
                label="Segmentos",
                interactive=False
            )
            gr.Markdown("### 🧹 Gestionar Segmentos")
            with gr.Row():
                seg_speaker = gr.Dropdown(choices=SPEAKERS, label="Hablante", value=SPEAKERS[0])
                seg_id_input = gr.Number(label="segment_id", value=1, precision=0)
            with gr.Row():
                seg_play_btn = gr.Button("▶️ Reproducir segmento")
                seg_delete_btn = gr.Button("🗑️ Eliminar segmento")
            seg_player = gr.Audio(label="Reproducción segmento")
            seg_action_msg = gr.Textbox(label="Acción", interactive=False)
    
    # Event handlers
    audio_input.change(
        fn=load_audio,
        inputs=[audio_input],
        outputs=[audio_player, load_msg, status_display]
    )
    
    preview_btn.click(
        fn=preview_segment,
        inputs=[start_h, start_m, start_s, end_h, end_m, end_s],
        outputs=[segment_player, preview_msg]
    )
    
    save_btn.click(
        fn=save_segment,
        inputs=[speaker_dropdown, start_h, start_m, start_s, end_h, end_m, end_s, notes_input],
        outputs=[save_msg, status_display]
    )

    emb_btn.click(
        fn=view_embeddings,
        inputs=[emb_speaker],
        outputs=[emb_plot, emb_info, emb_table]
    )

    seg_play_btn.click(
        fn=lambda spk, sid: manager.load_segment_audio(spk, sid),
        inputs=[seg_speaker, seg_id_input],
        outputs=[seg_player, seg_action_msg]
    )

    def _delete_seg(spk, sid):
        msg = manager.delete_segment(spk, sid)
        return msg, manager.get_speaker_status()

    seg_delete_btn.click(
        fn=_delete_seg,
        inputs=[seg_speaker, seg_id_input],
        outputs=[seg_action_msg, status_display]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  SISTEMA DE ENROLLMENT DE HABLANTES")
    print("="*60)
    print(f"\nDispositivo: {device}")
    print(f"Perfiles guardados en: {PROFILES_DIR.absolute()}")
    print("\nAbriendo interfaz web...")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
