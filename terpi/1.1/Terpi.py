import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
import json
import queue
import sys
import os
import time
import logging
import threading
import concurrent.futures
from functools import lru_cache
import traceback
import onnxruntime as ort
import numpy as np
import pyaudiowpatch as pyaudio
import tkinter.font as tkfont
from collections import deque
from ctypes import wintypes
from vosk import Model, KaldiRecognizer

try:
    from vosk import EndpointerMode
    _ENDPOINTER_DISPONIBLE = True
except ImportError:
    _ENDPOINTER_DISPONIBLE = False
    EndpointerMode = None

# ---------------------------------------------------------------------------
# Optimización de CTranslate2 (motor de Argos)
# ---------------------------------------------------------------------------
os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")
os.environ.setdefault("ARGOS_INTRA_THREADS", "2")
os.environ.setdefault("ARGOS_INTER_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CT2_FORCE_CPU_ISA", "AVX2")

import argostranslate.package
import argostranslate.translate
from punctuators.models import PunctCapSegModelONNX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
logging.getLogger("argostranslate").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Clase para SileroVAD
# ---------------------------------------------------------------------------

class SileroVAD:
    def __init__(self, ruta_modelo, sample_rate=16000):
        if sample_rate != 16000:
            raise ValueError("Silero VAD (v5) solo soporta 16kHz.")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = HILOS_SILERO_ONNX
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            ruta_modelo,
            providers=["CPUExecutionProvider"],
            sess_options=opts
        )
        self._sample_rate = np.array(sample_rate, dtype=np.int64)
        self._chunk_size = 512
        self._context_size = 64
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def process(self, audio_int16):
        if len(audio_int16) != self._chunk_size:
            raise ValueError(f"Se esperaban {self._chunk_size} muestras.")
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        audio_f32 = audio_f32.reshape(1, -1)
        input_with_context = np.concatenate([self._context, audio_f32], axis=1)
        self._context = audio_f32[:, -self._context_size:]
        output, self._state = self._session.run(
            None,
            {"input": input_with_context, "sr": self._sample_rate, "state": self._state}
        )
        return float(output[0][0])

# ---------------------------------------------------------------------------
# Manejo de excepciones fatales
# ---------------------------------------------------------------------------
def _manejar_excepcion_fatal(tipo, valor, tb):
    traceback.print_exception(tipo, valor, tb)
    print()
    print("=" * 60)
    print("Ami encontró un error y se cerrará.")
    print("Copia el mensaje de arriba y compártelo en el grupo.")
    print("=" * 60)
    try:
        input("Pulsa Enter para cerrar...")
    except Exception:
        pass

sys.excepthook = _manejar_excepcion_fatal


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    CARPETA_PROGRAMA = os.path.dirname(sys.executable)
else:
    CARPETA_PROGRAMA = os.path.dirname(os.path.abspath(__file__))

MODELO_ES = os.path.join(CARPETA_PROGRAMA, "vosk-model-es-0.42")
MODELO_EN = os.path.join(CARPETA_PROGRAMA, "vosk-model-en-us-0.22")
CARPETA_TRADUCCIONES = os.path.join(CARPETA_PROGRAMA, "translate models")

RUTA_APPDATA = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Terpi")
RUTA_CONFIG = os.path.join(RUTA_APPDATA, "config.json")


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MAX_COLA_AUDIO = 100
IDIOMA = "INGLÉS"
TRADUCCION_ACTIVA = True
PUNTUACION_ACTIVA = True
DEBUG_TEXTO = False
CONFIANZA_MINIMA = 0.3
PALABRAS_VACIAS = {"the", "a", "an", "of", "to", "in", "at", "on"}
MIN_PALABRAS_CONFIRMAR = 4
TIMEOUT_FLUSH_SEG = 0.75
BLOQUES_POR_CICLO = 5
TICK_PROCESAR_MS = 10
TICK_WATCHDOG_MS = 2000
TICK_BUFFER_MS = 100
TIMEOUT_STREAM_SEG = 5
CODIGO_ARGOS = {"INGLÉS": "en", "ESPAÑOL": "es"}
CACHE_TRADUCCIONES = 3000
MIN_PALABRAS_PUNTUAR = 4
HILOS_PROCESAMIENTO = 2
HILOS_CLASIFICACION = 1
HILOS_VOXLINGUA_ONNX = 2
HILOS_SILERO_ONNX = 1
VL_BUFFER_SEG = 2.0
VL_MIN_AUDIO_SEG = 2.0
VL_INTERVALO_SEG = 0.75
VL_CONFIANZA_MINIMA = 0.65

BUFFER_RECUPERACION_SEG = 5.0
SILENCIO_FIN_TURNO_SEG = 0.5

HOTKEY_ID = 1
VK_F8 = 0x77
WM_HOTKEY =0x0312
WM_QUIT = 0x0012
MOD_NOREPEAT = 0x4000

VERSION_TERPI = "1.1.0"
LICENCIA_TERPI = "GPL-3.0"

URL_GITHUB = "https://github.com/fizzyrox00-art/transcriptor-for-terps"
URL_GITHUB_ISSUES = "https://github.com/fizzyrox00-art/transcriptor-for-terps/issues"
URL_FACEBOOK = "https://www.facebook.com/santiago.cachocordova.3"
EMAIL_CONTACTO = "elvientoazul@outlook.com"

URL_PAYPAL = "https://www.paypal.com/donate/?hosted_button_id=EJE43CLNFE6Q6"
CLABE_DONACIONES = "722969010149372781"
BENEFICIARIO = "Ivan Santiago Cacho Cordova"

TEMAS_PREDEFINIDOS = {
    "Terpi Original": {
        "fondo_ventana": "#000000",
        "fondo_texto": "#1E1E1E",
        "color_parcial": "#B8B8B8",
        "color_ingles": "#a5d7f2",
        "color_español": "#f2b27a",
        "color_traduccion": "#999999",
        "fuente_texto": "Merriweather",
        "fuente_ui": "Jost",
        "color_boton": "#2A2A2A",
        "color_boton_fg": "#E0E0E0",
        "color_boton_active": "#3A3A3A",
        "tamaño_ui": 11,
    },
    "☀ Bubblegum": {
        "fondo_ventana": "#ffabcc",
        "fondo_texto": "#ffebf0",
        "color_parcial": "#5E5E5E",
        "color_ingles": "#008b8b",
        "color_español": "#4b0082",
        "color_traduccion": "#8B5A7A",
        "fuente_texto": "Comic Sans MS",
        "fuente_ui": "Segoe UI",
        "color_boton": "#f589a7",
        "color_boton_fg": "#2e0308",
        "color_boton_active": "#e37f9b",
        "tamaño_ui": 11,
    },
    "☀ Profesional": {
        "fondo_ventana": "#4e2a1b",
        "fondo_texto": "#ce8757",
        "color_parcial": "#5A3A2A",
        "color_ingles": "#000080",
        "color_español": "#cc0000",
        "color_traduccion": "#5A3A2A",
        "fuente_texto": "Georgia",
        "fuente_ui": "Segoe UI",
        "color_boton": "#402114",
        "color_boton_fg": "#f4ecd8",
        "color_boton_active": "#361f10",
        "tamaño_ui": 11,
    },
    "☀ Ocean": {
        "fondo_ventana": "#0b2e59",
        "fondo_texto": "#a8d8d8",
        "color_parcial": "#1A4A5A",
        "color_ingles": "#3a8756",
        "color_español": "#293d85",
        "color_traduccion": "#1A4A5A",
        "fuente_texto": "Consolas",
        "fuente_ui": "Segoe UI",
        "color_boton": "#1b5a8a",
        "color_boton_fg": "#ffffff",
        "color_boton_active": "#2470a8",
        "tamaño_ui": 11,
    },
    "☀ TerpiAmiga": {
        "fondo_ventana": "#0055AA",
        "fondo_texto": "#AAAAAA",
        "color_parcial": "#383838",
        "color_ingles": "#000000",
        "color_español": "#000000",
        "color_traduccion": "#1c1c1c",
        "fuente_texto": "Topaz a500a1000a2000",
        "fuente_ui": "Topaz a500a1000a2000",
        "color_boton": "#AAAAAA",
        "color_boton_fg": "#000000",
        "color_boton_active": "#0055AA",
        "tamaño_ui": 11,
    },
    "🌙 Midnight": {
        "fondo_ventana": "#0a0a12",
        "fondo_texto": "#12121a",
        "color_parcial": "#B8B8B8",
        "color_ingles": "#66b2ff",
        "color_español": "#ffb366",
        "color_traduccion": "#999999",
        "fuente_texto": "Consolas",
        "fuente_ui": "Segoe UI",
        "color_boton": "#1e2a3a",
        "color_boton_fg": "#c8d8e8",
        "color_boton_active": "#2a3a50",
        "tamaño_ui": 11,
    },
    "🌙 Cyberpunk": {
        "fondo_ventana": "#000000",
        "fondo_texto": "#111111",
        "color_parcial": "#B8B8B8",
        "color_ingles": "#00ffcc",
        "color_español": "#ff00ff",
        "color_traduccion": "#732f2f",
        "fuente_texto": "Consolas",
        "fuente_ui": "Segoe UI",
        "color_boton": "#003d33",
        "color_boton_fg": "#00ffcc",
        "color_boton_active": "#006655",
        "tamaño_ui": 11,
    },
    "🌙 Sakura Night": {
        "fondo_ventana": "#1a1a24",
        "fondo_texto": "#222230",
        "color_parcial": "#B8B8B8",
        "color_ingles": "#ff99cc",
        "color_español": "#99ccff",
        "color_traduccion": "#684269",
        "fuente_texto": "Merriweather",
        "fuente_ui": "Segoe UI",
        "color_boton": "#3a2a3a",
        "color_boton_fg": "#ffb3d9",
        "color_boton_active": "#4d3a4d",
        "tamaño_ui": 11,
    },
    "🌙 Cassette Futurism": {
        "fondo_ventana": "#000000",
        "fondo_texto": "#000000",
        "color_parcial": "#B8B8B8",
        "color_ingles": "#207a39",
        "color_español": "#C17C1A",
        "color_traduccion": "#4A8A5A",
        "fuente_texto": "VT323",
        "fuente_ui": "VT323",
        "color_boton": "#000000",
        "color_boton_fg": "#C9D9D8",
        "color_boton_active": "#01a3b2",
        "tamaño_ui": 14,
    }
}

_config_visual = {
    "fondo_ventana": "#000000",
    "fondo_texto": "#1E1E1E",
    "fuente_texto": "Merriweather",
    "color_ingles": "#a5d7f2",
    "color_español": "#f2b27a",
    "color_traduccion": "#999999",
    "color_parcial": "#B8B8B8",
    "fuente_ui": "Jost",
    "modo_dislexia": False,
    "fuente_dislexia": "OpenDyslexic",
    "tamaño_texto": 11,
    "tamaño_ui": 11,
    "modo_tdah": False,
    "modo_daltonismo_rv": False,
    "modo_daltonismo_aa": False,
    "modo_vision_cansada": False,
    "imagen_fondo": None,
    "color_boton": "#2a2a2a",
    "color_boton_fg": "white",
    "color_boton_active": "#3a3a3a",
    "geometria_ventana": "900x650+100+50",
    "deteccion_automatica_idioma": True,
}


# ---------------------------------------------------------------------------
# Carga y guardado de configuración
# ---------------------------------------------------------------------------
def cargar_configuracion():
    global _config_visual
    if os.path.exists(RUTA_CONFIG):
        try:
            with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
                datos_guardados = json.load(f)
                _config_visual.update(datos_guardados)
            log.info("Configuración cargada desde %s", RUTA_CONFIG)
        except Exception as e:
            log.error("Error al cargar la configuración: %s", e)

    if "color_ingles" not in _config_visual:
        _config_visual["color_ingles"] = "#a5d7f2"
    if "color_español" not in _config_visual:
        _config_visual["color_español"] = "#f2b27a"

    _config_visual.setdefault("color_traduccion", "#999999")
    _config_visual.setdefault("color_parcial", "#B8B8B8")
    _config_visual.setdefault("color_boton", "#2A2A2A")
    _config_visual.setdefault("color_boton_fg", "white")
    _config_visual.setdefault("color_boton_active", "#3A3A3A")
    _config_visual.setdefault("geometria_ventana", "900x650+100+50")
    _config_visual.setdefault("tamaño_ui", 11)
    _config_visual.setdefault("deteccion_automatica_idioma", True)


def guardar_configuracion():
    try:
        os.makedirs(RUTA_APPDATA, exist_ok=True)
        with open(RUTA_CONFIG, "w", encoding="utf-8") as f:
            json.dump(_config_visual, f, indent=4, ensure_ascii=False)
    except Exception as e:
        log.error("Error al guardar la configuración: %s", e)


cargar_configuracion()


# ---------------------------------------------------------------------------
# Variables globales de la UI
# ---------------------------------------------------------------------------
frame_fuente_ref = None
frame_toggles_ref = None
frame_panel_superior = None
combo_fuente = None
btn_reiniciar = None
var_traduccion = None
chk_traduccion = None
var_puntuacion = None
chk_puntuacion = None
label_estado = None
texto = None
scroll_texto = None
preview_widget = None
label_fondo = None
btn_mostrar_flotante = None
ventana_ajustes_ref = None
_widgets_fuente_ui = []
_imagen_fondo_original = None
_job_fondo = None
_ultimo_tamaño_fondo = (0, 0)
_info_completa_ref = None
titulo_carga = None
mensaje_carga = None
gif_carga = None


# ---------------------------------------------------------------------------
# Ventana principal y Pantalla de Carga
# ---------------------------------------------------------------------------
ventana = tk.Tk()

_ruta_ico = os.path.join(CARPETA_PROGRAMA, "assets", "Terpicono.ico")
if os.path.exists(_ruta_ico):
    try:
        ventana.iconbitmap(_ruta_ico)
    except Exception as e:
        log.warning("No se pudo cargar el icono: %s", e)

_geo = _config_visual.get("geometria_ventana", "900x650+100+50")
try:
    if "x" in _geo and ("+" in _geo or "-" in _geo):
        ventana.geometry(_geo)
    else:
        ventana.geometry("900x650")
except Exception:
    ventana.geometry("900x650")

ventana.configure(bg=_config_visual["fondo_ventana"])
ventana.title("[EN] Terpi")

def _cargar_gif_animado(ruta_gif, label, intervalo_ms=None, tamaño_max=(300, 300)):
    if not os.path.exists(ruta_gif):
        log.warning("GIF no encontrado: %s", ruta_gif)
        return
    try:
        from PIL import Image, ImageTk, ImageSequence

        frames = []
        delays = []
        img = Image.open(ruta_gif)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGBA")
            if tamaño_max is not None:
                frame.thumbnail(tamaño_max, Image.Resampling.LANCZOS)
            frames.append(ImageTk.PhotoImage(frame))
            delays.append(frame.info.get("duration", intervalo_ms or 100))

        if not frames:
            return

        log.info("GIF cargado: %s (%d frames)",
                 os.path.basename(ruta_gif),
                 len(frames))

        def _animar(indice=0):
            try:
                if not label.winfo_exists():
                    return
                label.config(image=frames[indice])
                label.image = frames[indice]
                delay = delays[indice] if intervalo_ms is None else intervalo_ms
                label.after(max(10, delay), _animar, (indice + 1) % len(frames))
            except Exception:
                pass

        _animar()

    except Exception as e:
        log.warning("No se pudo cargar el GIF animado: %s", e)

_ruta_gif_carga = os.path.join(CARPETA_PROGRAMA, "assets", "ami_carga.gif")
gif_carga = tk.Label(ventana, bg=_config_visual["fondo_ventana"])
gif_carga.pack(pady=(120, 20))
_cargar_gif_animado(_ruta_gif_carga, gif_carga, tamaño_max=(300, 300))

titulo_carga = tk.Label(
    ventana, text="TERPI",
    font=("Jost", 40),
    bg=_config_visual["fondo_ventana"],
    fg="white"
)
titulo_carga.pack(pady=(10, 10))

mensaje_carga = tk.Label(
    ventana, text="Terpi está iniciando...",
    font=("Jost", 14),
    bg=_config_visual["fondo_ventana"],
    fg="#BBBBBB"
)
mensaje_carga.pack()
ventana.update()

def _validar_geometria_en_pantalla():
    try:
        ventana.update_idletasks()
        x = ventana.winfo_x()
        y = ventana.winfo_y()
        w = ventana.winfo_width()
        h = ventana.winfo_height()
        sw = ventana.winfo_screenwidth()
        sh = ventana.winfo_screenheight()

        fuera_x = x + w < 0 or x > sw
        fuera_y = y + h < 0 or y > sh
        if fuera_x or fuera_y:
            ventana.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    except Exception as e:
        log.warning("Error validando geometría: %s", e)

import ctypes

def _cargar_fuente_local(ruta_relativa):
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    
    ruta = os.path.join(base, ruta_relativa)
    
    if not os.path.exists(ruta):
        log.warning("Fuente no encontrada: %s", ruta)
        return False
        
    try:
        add_font = ctypes.windll.gdi32.AddFontResourceExW
        path_buf = ctypes.create_unicode_buffer(ruta)
        if add_font(path_buf, 0x10, 0):
            log.info("Fuente cargada: %s", os.path.basename(ruta))
            return True
    except Exception as e:
        log.warning("Error cargando fuente %s: %s", ruta, e)
    return False


_cargar_fuente_local("assets/fonts/Jost-Regular.ttf")
_cargar_fuente_local("assets/fonts/Jost-Bold.ttf")
_cargar_fuente_local("assets/fonts/Topaz-a500.ttf")
_cargar_fuente_local("assets/fonts/VT323.ttf")
_cargar_fuente_local("assets/fonts/Merriweather-Regular.ttf")
_cargar_fuente_local("assets/fonts/Merriweather-Italic.ttf")
_cargar_fuente_local("assets/fonts/OpenDyslexic-Regular.otf")
_cargar_fuente_local("assets/fonts/OpenDyslexic-Bold.otf")
_cargar_fuente_local("assets/fonts/AtkinsonHyperlegible-Bold.ttf")
_cargar_fuente_local("assets/fonts/AtkinsonHyperlegible-Italic.ttf")
_cargar_fuente_local("assets/fonts/AtkinsonHyperlegible-Regular.ttf")
log.info("Todas las fuentes cargadas. Familias disponibles: %s", 
         [f for f in tkfont.families() if "Dyslexic" in f or "Jost" in f or "Topaz" in f])


ventana.after(100, _validar_geometria_en_pantalla)


# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_modelos = {"ESPAÑOL": None, "INGLÉS": None}
_reconocedores = {"ESPAÑOL": None, "INGLÉS": None}
SR_ACTUAL = 48000
_puntuador = None
RUTAS_MODELOS = {"ESPAÑOL": MODELO_ES, "INGLÉS": MODELO_EN}
_traducciones = {}
historial = []
_buffer_frase = ""
_ultimo_flush = time.time()
_idioma_buffer = None
_buffer_recuperacion = deque()
_bytes_recuperacion = 0

_generacion_clasificador = 0

_turno_en_validacion = False
_textos_provisionales = []

_muestras_silencio = 0

_ultimo_parcial = ""
audio_queue = queue.Queue(maxsize=MAX_COLA_AUDIO)
cola_ui = queue.Queue()
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=HILOS_PROCESAMIENTO, thread_name_prefix="PostProcesamiento")
_executor_voxlingua = concurrent.futures.ThreadPoolExecutor(max_workers=HILOS_CLASIFICACION, thread_name_prefix="VoxLingua")
stream_actual = {
    "stream": None,
    "pyaudio": None,
    "device": None,
    "sr": None,
    "canales": 1,
    "ultimo_bloque": time.time()
}
_dispositivos_cache = []
_primera_carga = True
_estadisticas_cache = {"hits": 0, "misses": 0}

_sesion_voxlingua = None
_mapa_idiomas = {}
_buffer_idioma = np.array([], dtype=np.int16)
_ultimo_intento_clasificar = 0.0

_vad_silero = None
_buffer_vad = np.array([], dtype=np.int16)
_umbral_voz = 0.5
_clasificando_idioma = False
_en_intervencion = False

_hotkey_registrado = False
_hotkey_thread = None
_hotkey_thread_id = None



# ---------------------------------------------------------------------------
# Lógica de procesamiento
# ---------------------------------------------------------------------------
def _normalizar_palabra(p):
    return p.strip(".,?!;:\"-—–()[]{}«»“”‘’¿¡").lower()

def iniciar_turno_provisional():
    global _turno_en_validacion
    global _buffer_idioma
    global _generacion_clasificador
    global _ultimo_intento_clasificar
    _generacion_clasificador += 1
    _flush_buffer(forzar=True)
    _turno_en_validacion = True
    _textos_provisionales.clear()
    limpiar_audio_recuperacion()
    _buffer_idioma = np.array([], dtype=np.int16)
    _ultimo_intento_clasificar = 0.0

    log.info("Inicio de intervención provisional")

def confirmar_turno_provisional():
    global _turno_en_validacion
    _turno_en_validacion = False
    for contenido in _textos_provisionales:
        _confirmar_con_buffer(contenido)
    _textos_provisionales.clear()

    limpiar_audio_recuperacion()
    actualizar_texto_parcial()

def puntuar(texto_original):
    if not PUNTUACION_ACTIVA or _puntuador is None:
        return texto_original
    if not texto_original.strip():
        return texto_original
    if len(texto_original.split()) < MIN_PALABRAS_PUNTUAR:
        return texto_original

    try:
        resultados = _puntuador.infer([texto_original])
        if not resultados or not resultados[0]:
            return texto_original
        texto_final = " ".join(resultados[0]).strip()
    except Exception as e:
        log.warning("Error en puntuación: %s", e)
        return texto_original

    if not texto_final:
        return texto_original

    palabras_original = set(_normalizar_palabra(w) for w in texto_original.split())
    palabras_final = set(_normalizar_palabra(w) for w in texto_final.split())
    perdidas = palabras_original - palabras_final
    if perdidas:
        log.debug("Puntuador perdió palabras: %s. Usando original.", perdidas)
        return texto_original

    return texto_final


@lru_cache(maxsize=CACHE_TRADUCCIONES)
def _traducir_cacheado(texto_original, from_code, to_code):
    traductor = _traducciones.get((from_code, to_code))
    if traductor is None:
        return None
    try:
        return traductor.translate(texto_original)
    except Exception as e:
        log.warning("Error traduciendo: %s", e)
        return None


def traducir(texto_original, idioma_origen):
    if not TRADUCCION_ACTIVA:
        return None
    from_code = CODIGO_ARGOS.get(idioma_origen)
    if from_code not in ("en", "es"):
        return None
    to_code = "es" if from_code == "en" else "en"
    info_antes = _traducir_cacheado.cache_info()
    resultado = _traducir_cacheado(texto_original, from_code, to_code)
    info_despues = _traducir_cacheado.cache_info()
    if info_despues.hits > info_antes.hits:
        _estadisticas_cache["hits"] += 1
    else:
        _estadisticas_cache["misses"] += 1

    return resultado


def listar_loopbacks_disponibles():
    resultado = []
    try:
        with pyaudio.PyAudio() as p:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output = wasapi_info.get("defaultOutputDevice", -1)

            for loopback in p.get_loopback_device_info_generator():
                nombre_altavoz = loopback["name"]
                nombre_visible = nombre_altavoz.replace(" [Loopback]", "").strip()

                es_defecto = False
                try:
                    default_speakers = p.get_device_info_by_index(default_output)
                    if default_speakers["name"] in nombre_altavoz:
                        es_defecto = True
                except Exception:
                    pass

                resultado.append({
                    "indice": loopback["index"],
                    "nombre": nombre_visible,
                    "sr": int(loopback["defaultSampleRate"]),
                    "canales": min(loopback["maxInputChannels"], 2),
                    "es_defecto": es_defecto,
                })
    except Exception as e:
        log.error("Error listando dispositivos de loopback: %s", e)

    resultado.sort(key=lambda d: not d["es_defecto"])
    return resultado

def _remuestrear_a_16k(audio_np, sr_original):

    if sr_original == 16000:
        return audio_np.astype(np.int16)
    try:
        n_salida = int(len(audio_np) * 16000 / sr_original)
        indices = np.linspace(0, len(audio_np) - 1, n_salida)
        audio_16k = np.interp(indices, np.arange(len(audio_np)), audio_np)
        return audio_16k.astype(np.int16)
    except Exception as e:
        log.warning("Error remuestreando audio: %s", e)
        return audio_np


def preparar_reconocedores(sr):
    global SR_ACTUAL
    SR_ACTUAL = sr

    for idioma, modelo in _modelos.items():
        if modelo is None:
            continue
        rec = KaldiRecognizer(modelo, sr)
        rec.SetWords(True)

        if _ENDPOINTER_DISPONIBLE:
            try:
                rec.SetEndpointerMode(EndpointerMode.LONG)
            except Exception as e:
                log.warning("No se pudo activar EndpointerMode.LONG: %s", e)

        _reconocedores[idioma] = rec


def flush_reconocedor_actual():
    rec = _reconocedores.get(IDIOMA)
    if rec is None:
        return ""
    try:
        rec.AcceptWaveform(b"")
        resultado = json.loads(rec.FinalResult())
        return resultado.get("text", "").strip()
    except Exception as e:
        log.warning("Error en flush_reconocedor_actual: %s", e)
        return ""


def callback_pyaudio(in_data, frame_count, time_info, status):
    if status:
        log.warning("Audio status: %s", status)
    stream_actual["ultimo_bloque"] = time.time()
    try:
        audio_np = np.frombuffer(in_data, dtype=np.int16)
        canales = stream_actual.get("canales", 1)
        if canales > 1:
            audio_np = (audio_np.reshape(-1, canales).mean(axis=1).astype(np.int16))
        audio_queue.put_nowait(audio_np.tobytes())
    except queue.Full:
        pass
    except Exception as e:
        log.warning("Error en callback: %s", e)
    return (None, pyaudio.paContinue)


def cerrar_stream():
    s = stream_actual.get("stream")
    if s is not None:
        try:
            s.stop_stream()
            s.close()
        except Exception as e:
            log.warning("Error cerrando stream: %s", e)
    p = stream_actual.get("pyaudio")
    if p is not None:
        try:
            p.terminate()
        except Exception as e:
            log.warning("Error cerrando PyAudio: %s", e)
    stream_actual["stream"] = None
    stream_actual["pyaudio"] = None
    stream_actual["device"] = None
    stream_actual["sr"] = None


def abrir_stream(dispositivo_dict):
    cerrar_stream()
    _flush_buffer(forzar=True)
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    indice = dispositivo_dict["indice"]
    sr = dispositivo_dict["sr"]
    canales = dispositivo_dict.get("canales", 2)
    stream_actual["canales"] = canales
    preparar_reconocedores(sr)

    try:
        p = pyaudio.PyAudio()
        blocksize = int(sr * 0.1)
        stream = p.open(
            format=pyaudio.paInt16, channels=canales, rate=sr,
            frames_per_buffer=blocksize, input=True,
            input_device_index=indice, stream_callback=callback_pyaudio,
        )
        stream.start_stream()
        stream_actual["stream"] = stream
        stream_actual["pyaudio"] = p
        stream_actual["device"] = indice
        stream_actual["sr"] = sr
        stream_actual["ultimo_bloque"] = time.time()
    except Exception as e:
        log.error("Error abriendo stream: %s", e)
        raise


def _etiqueta_y_prefijo(idioma):
    if idioma == "INGLÉS":
        return "ingles", "Rep: "
    return "español", "Lep: "


def _sufijo_traduccion(idioma_origen):
    return "ES" if idioma_origen == "INGLÉS" else "EN"


def _renderizar_todo():
    texto.config(state="normal")
    texto.delete("1.0", "end")
    for entrada in historial:
        texto.insert("end", entrada["original"] + "\n", entrada["etiqueta"])
        if entrada.get("traduccion"):
            texto.insert("end",
                         entrada["traduccion"] + "\n",
                         "traduccion")
        texto.insert("end",
                     "\n")
    texto.insert("end",
                 "\n")
    texto.mark_set("parcial_inicio",
                   "end-1c")
    texto.mark_gravity("parcial_inicio",
                       tk.LEFT)
    texto.config(state="disabled")
    texto.see("end")


def mostrar_parcial(contenido):
    texto.config(state="normal")
    texto.delete("parcial_inicio", "end")
    if contenido:
        caracter_anterior = texto.get("parcial_inicio-1c",
                                      "parcial_inicio")
        if caracter_anterior != "\n" and texto.index("parcial_inicio") != "1.0":
            texto.insert("parcial_inicio", "\n")
            texto.mark_set("parcial_inicio", "end-1c")
        _, prefijo = _etiqueta_y_prefijo(IDIOMA)
        texto.insert("end",
                     f"{prefijo}{contenido}…",
                     "parcial")
    texto.config(state="disabled")
    texto.see("end")

def actualizar_texto_parcial():
    partes = []

    if _buffer_frase:
        partes.append(_buffer_frase)
    if _turno_en_validacion:
        partes.extend(_textos_provisionales)
    if _ultimo_parcial:
        partes.append(_ultimo_parcial)

    contenido = " ".join(partes).strip()

    mostrar_parcial(contenido)

def mostrar_final(contenido):
    _, prefijo = _etiqueta_y_prefijo(IDIOMA)
    etiqueta_orig = "ingles" if IDIOMA == "INGLÉS" else "español"
    texto.config(state="normal")
    texto.delete("parcial_inicio", "end")
    if texto.index("end-1c") != "1.0":
        texto.insert("end",
                     f"\n{prefijo}{contenido}\n\n",
                     etiqueta_orig)
    else:
        texto.insert("end",
                     f"{prefijo}{contenido}\n\n",
                     etiqueta_orig)
    texto.mark_set("parcial_inicio", "end-1c")
    texto.mark_gravity("parcial_inicio", tk.LEFT)
    
    texto.config(state="disabled")
    texto.see("end")
    entrada = {
        "original": prefijo + contenido,
        "etiqueta": etiqueta_orig,
        "traduccion": None,
        "idioma_origen": IDIOMA,
        "contenido_crudo": contenido,
    }
    historial.append(entrada)
    _executor.submit(_procesar_entrada, entrada)


def _procesar_entrada(entrada):
    try:
        contenido = entrada["contenido_crudo"]
        idioma_origen = entrada["idioma_origen"]
        texto_puntuado = puntuar(contenido)
        trad = traducir(texto_puntuado, idioma_origen)

    except Exception as e:
        log.warning("Error procesando entrada: %s", e)
        return

    cola_ui.put((
        "aplicar_resultado", entrada, texto_puntuado, trad))


def _aplicar_resultado(entrada, texto_puntuado, trad):
    _, prefijo = _etiqueta_y_prefijo(entrada["idioma_origen"])
    entrada["original"] = prefijo + texto_puntuado
    if trad:
        sufijo = _sufijo_traduccion(entrada["idioma_origen"])
        entrada["traduccion"] = f"{' ' * 5}{sufijo}: {trad}"
    
    forzar_redibujado_texto()


def _confirmar_con_buffer(texto_nuevo):
    global _buffer_frase, _ultimo_flush, _idioma_buffer
    if _idioma_buffer is not None and _idioma_buffer != IDIOMA:
        _flush_buffer(forzar=True)
    _idioma_buffer = IDIOMA
    if _buffer_frase:
        _buffer_frase = (_buffer_frase + " " + texto_nuevo).strip()
    else:
        _buffer_frase = texto_nuevo.strip()
    _ultimo_flush = time.time()
    palabras = _buffer_frase.split()
    tiene_punto = any(_buffer_frase.rstrip().endswith(p) for p in ".?!")
    if len(palabras) >= MIN_PALABRAS_CONFIRMAR or tiene_punto:
        contenido = _buffer_frase
        _buffer_frase = ""
        mostrar_final(contenido)
    else:
        mostrar_parcial(_buffer_frase)


def _flush_buffer(forzar=False):
    global _buffer_frase, _ultimo_flush
    if not _buffer_frase:
        return
    if not forzar and (time.time() - _ultimo_flush) < TIMEOUT_FLUSH_SEG:
        return
    contenido = _buffer_frase
    _buffer_frase = ""
    _ultimo_flush = time.time()
    if contenido.strip():
        mostrar_final(contenido)


def revisar_buffer():
    try:
        _flush_buffer(forzar=False)
    except Exception as e:
        log.exception("Error en revisar_buffer: %s", e)
    finally:
        ventana.after(TICK_BUFFER_MS, revisar_buffer)


def es_alucinacion(texto_final, resultado_json):
    if not texto_final:
        return True
    palabras = texto_final.split()
    confs = [w.get("conf", 0.0) for w in resultado_json.get("result", [])]
    conf_prom = sum(confs) / len(confs) if confs else 1.0
    if conf_prom < CONFIANZA_MINIMA:
        return True
    if len(palabras) == 1 and palabras[0].lower() in PALABRAS_VACIAS:
        return True
    return False


def _aplicar_dispositivos(dispositivos):
    global _dispositivos_cache, _primera_carga
    _dispositivos_cache = dispositivos

    etiquetas = []
    for d in dispositivos:
        marca = " ⭐" if d["es_defecto"] else ""
        etiquetas.append(f"🔊 [{d['indice']:3d}] {d['nombre']}{marca}  ({d['sr']} Hz, {d['canales']} canales)")
    combo_fuente["values"] = etiquetas

    if not dispositivos:
        label_estado.config(text="❌ No se encontraron dispositivos de audio (WASAPI).")
        return

    if _primera_carga:
        _primera_carga = False
        mejor = dispositivos[0]
        combo_fuente.current(0)
        label_estado.config(text=f"✅ {mejor['nombre']} @ {mejor['sr']} Hz")
        try:
            abrir_stream(mejor)
        except Exception as e:
            log.error("Error abriendo stream inicial: %s", e)
            label_estado.config(text=f"❌ Error: {e}")
    else:
        label_estado.config(text=f"🔄 Lista actualizada ({len(dispositivos)} dispositivos).")


def refrescar_lista_dispositivos():
    label_estado.config(text="⏳ Buscando dispositivos de audio... (Espere un segundo)")
    ventana.update()

    def tarea():
        try:
            dispositivos = listar_loopbacks_disponibles()
        except Exception as e:
            log.exception("Error escaneando: %s", e)
            dispositivos = []
        ventana.after(0, lambda: _aplicar_dispositivos(dispositivos))

    threading.Thread(target=tarea, daemon=True).start()


def al_cambiar_fuente(event=None):
    sel = combo_fuente.current()
    if sel < 0 or sel >= len(_dispositivos_cache):
        return
    d = _dispositivos_cache[sel]
    try:
        abrir_stream(d)
        label_estado.config(text=f"✅ {d['nombre']} @ {d['sr']} Hz")
    except Exception as e:
        log.error("Error cambiando de fuente: %s", e)
        label_estado.config(text=f"❌ Error: {e}")



def cambiar_idioma(event):
    global IDIOMA, _ultimo_parcial
    texto_pendiente = flush_reconocedor_actual()
    _flush_buffer(forzar=True)

    rec_actual = _reconocedores.get(IDIOMA)
    if rec_actual is not None:
        rec_actual.Reset()

    if texto_pendiente:
        _confirmar_con_buffer(texto_pendiente)
        _flush_buffer(forzar=True)
    _ultimo_parcial = ""

    if IDIOMA == "INGLÉS":
        IDIOMA = "ESPAÑOL"
        ventana.title("[ES] Terpi")
    else:
        IDIOMA = "INGLÉS"
        ventana.title("[EN] Terpi")

    texto.config(state="normal")
    texto.delete("parcial_inicio", "end")
    texto.config(state="disabled")

def _ruta_asset(relativa):
    ruta_externa = os.path.join(CARPETA_PROGRAMA, relativa)
    if os.path.exists(ruta_externa):
        return ruta_externa
    if getattr(sys, "frozen", False):
        ruta_bundle = os.path.join(sys._MEIPASS, relativa)
        if os.path.exists(ruta_bundle):
            return ruta_bundle
    return ruta_externa

def _cargar_vad():
    global _vad_silero
    ruta = _ruta_asset("assets/voice recon models/silero_vad.onnx")
    if not os.path.exists(ruta):
        log.warning("Silero VAD no encontrado: %s", ruta)
        return
    try:
        _vad_silero = SileroVAD(ruta)
        log.info("Silero VAD cargado")
    except Exception as e:
        log.error("Error cargando Silero VAD: %s", e)
        _vad_silero = None

def _cargar_voxlingua():
    global _sesion_voxlingua, _mapa_idiomas
    if _sesion_voxlingua is not None:
        return
    ruta_modelo = _ruta_asset(os.path.join("assets", "voice recon models", "voxlingua107.onnx"))
    ruta_labels = _ruta_asset(os.path.join("assets", "voice recon models", "lang_map.json"))
    if not os.path.exists(ruta_modelo):
        log.warning("Modelo VoxLingua107 no encontrado: %s", ruta_modelo)
        return
    try:
        opts = ort.SessionOptions()

        opts.intra_op_num_threads = HILOS_VOXLINGUA_ONNX
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

        _sesion_voxlingua = ort.InferenceSession(ruta_modelo, sess_options=opts, providers=["CPUExecutionProvider"])
        if os.path.exists(ruta_labels):
            with open(ruta_labels, "r", encoding="utf-8") as f:
                _mapa_idiomas = json.load(f)
        else:
            log.warning("Labels de VoxLingua107 no encontrados: %s", ruta_labels)
        log.info("VoxLingua107 cargado (%d idiomas)", len(_mapa_idiomas))
    except Exception as e:
        log.error("Error cargando VoxLingua107: %s", e)
        _sesion_voxlingua = None


def _clasificar_idioma(audio_np):
    if _sesion_voxlingua is None:
        _cargar_voxlingua()
        if _sesion_voxlingua is None:
            return None, 0.0
    try:
        if audio_np.dtype != np.float32:
            audio_float = audio_np.astype(np.float32) / 32768.0
        else:
            audio_float = audio_np
        input_name = _sesion_voxlingua.get_inputs()[0].name
        salida = _sesion_voxlingua.run(None, {input_name: audio_float.reshape(1, -1)})[0]

        logits = salida[0]
        logits_estables = logits - np.max(logits)
        exp_logits = np.exp(logits_estables)
        probabilidades = exp_logits / np.sum(exp_logits)
        
        indice = int(np.argmax(salida))
        confianza = float(probabilidades[indice])
        
        entrada = _mapa_idiomas.get(str(indice), _mapa_idiomas.get(indice, {}))
        if isinstance(entrada, dict):
            idioma = entrada.get("iso", "unknown")
        else:
            idioma = entrada
        return idioma, confianza
    except Exception as e:
        log.warning("Error clasificando idioma: %s", e)
        return None, 0.0


def _tarea_clasificar_idioma(audio_np, generacion):
    global _clasificando_idioma

    try:
        inicio = time.perf_counter()
        idioma, confianza = _clasificar_idioma(audio_np)
        duracion = (
            time.perf_counter() - inicio
        ) * 1000
        log.info("VoxLingua: idioma=%s | confianza=%.2f | " "inferencia=%.1f ms", idioma, confianza, duracion)

        cola_ui.put(("resultado_clasificacion", idioma, confianza, generacion))

    except Exception:
        log.exception("Error en VoxLingua")

    finally:
        _clasificando_idioma = False


def _cambiar_idioma_por_clasificador(idioma_detectado):
    global IDIOMA
    nuevo = "INGLÉS" if idioma_detectado == "en" else "ESPAÑOL"
    if nuevo == IDIOMA:
        return
    cambiar_idioma(None)

def gestionar_resultado_clasificacion(idioma, confianza):
    if idioma not in ("en", "es"):
        return
    if confianza < VL_CONFIANZA_MINIMA:
        return
    nuevo_idioma = ("INGLÉS" if idioma == "en" else "ESPAÑOL")

    if nuevo_idioma == IDIOMA:
        if _turno_en_validacion:
            confirmar_turno_provisional()
    else:
        corregir_idioma_automatico(nuevo_idioma)

def corregir_idioma_automatico(nuevo_idioma):
    global IDIOMA, _ultimo_parcial
    global _turno_en_validacion
    global _buffer_idioma
    global _generacion_clasificador

    rec_nuevo = _reconocedores.get(nuevo_idioma)

    if rec_nuevo is None:
        return
    bloques = list(_buffer_recuperacion)
    log.info("Correción automática: %s ➡ %s" "Audio recuperado: %.2f segundos", IDIOMA, nuevo_idioma, sum(len(b) for b in bloques) / (SR_ACTUAL * 2))
    _generacion_clasificador += 1
    _flush_buffer(forzar=True)
    rec_anterior = _reconocedores.get(IDIOMA)

    if rec_anterior is not None:
        rec_anterior.Reset()

    _textos_provisionales.clear()
    _ultimo_parcial = ""
    mostrar_parcial("")
    IDIOMA = nuevo_idioma

    ventana.title("[EN] Terpi" if IDIOMA == "INGLÉS" else "[ES] Terpi")

    rec_nuevo.Reset()
    _turno_en_validacion = False
    _buffer_idioma = np.array([], dtype=np.int16)
    limpiar_audio_recuperacion()

    for bloque in bloques:
        procesar_bloque_vosk(rec_nuevo, bloque)
        



def limpiar_conversacion(event):
    global historial, _buffer_frase, _ultimo_parcial
    historial = []
    _buffer_frase = ""
    _ultimo_parcial = ""
    for rec in _reconocedores.values():
        if rec is not None:
            rec.Reset()
    _renderizar_todo()


def alternar_traduccion_teclado(event=None):
    var_traduccion.set(not var_traduccion.get())
    alternar_traduccion()


def alternar_puntuacion_teclado(event=None):
    var_puntuacion.set(not var_puntuacion.get())
    alternar_puntuacion()

def guardar_audio_recuperacion(data):
    global _bytes_recuperacion
    _buffer_recuperacion.append(data)
    _bytes_recuperacion += len(data)

    max_bytes = int(SR_ACTUAL * 2 * BUFFER_RECUPERACION_SEG)
    while (_bytes_recuperacion > max_bytes and _buffer_recuperacion):
        bloque = _buffer_recuperacion.popleft()
        _bytes_recuperacion -= len(bloque)

def limpiar_audio_recuperacion():
    global _bytes_recuperacion
    _buffer_recuperacion.clear()
    _bytes_recuperacion = 0

def establecer_deteccion_automatica(activo):
    global _generacion_clasificador
    global _buffer_idioma, _buffer_vad
    global _ultimo_intento_clasificar
    global _turno_en_validacion
    global _muestras_silencio

    _generacion_clasificador += 1
    _config_visual["deteccion_automatica_idioma"] = activo
    _buffer_idioma = np.array([], dtype=np.int16)
    _buffer_vad = np.array([], dtype=np.int16)
    _ultimo_intento_clasificar = 0.0
    _muestras_silencio = 0

    limpiar_audio_recuperacion()

    if _vad_silero is not None:
        _vad_silero.reset()

    if not activo:
        _turno_en_validacion = False
        for contenido in _textos_provisionales:
            _confirmar_con_buffer(contenido)
        _textos_provisionales.clear()

    guardar_configuracion()
    log.info("Detección automática: %s", "ON" if activo else "OFF")

def procesar_bloque_vosk(rec, data):
    global _ultimo_parcial

    if rec.AcceptWaveform(data):
        resultado = json.loads(rec.Result())
        texto_final = resultado.get("text", "").strip()
        _ultimo_parcial = ""

        if texto_final and not es_alucinacion(texto_final, resultado):
            recibir_texto_vosk(texto_final)

    else:
        parcial = json.loads(rec.PartialResult())
        _ultimo_parcial = parcial.get("partial", "")

    actualizar_texto_parcial()             


def procesar_audio():
    global _buffer_idioma, _ultimo_intento_clasificar
    global _buffer_vad, _clasificando_idioma
    global _muestras_silencio, _en_intervencion

    try:
        for _ in range(BLOQUES_POR_CICLO):
            rec = _reconocedores.get(IDIOMA)
            if rec is None:
                break

            try:
                data = audio_queue.get_nowait()
            except queue.Empty:
                break

            modo_auto = (
                _config_visual.get("deteccion_automatica_idioma", True) and _sesion_voxlingua is not None and _vad_silero is not None)

            if modo_auto:
                audio_np = np.frombuffer(data, dtype=np.int16)
                audio_16k = _remuestrear_a_16k(audio_np, SR_ACTUAL)
                _buffer_vad = np.concatenate([_buffer_vad, audio_16k])

                while len(_buffer_vad) >= 512:
                    chunk_vad = _buffer_vad[:512]
                    _buffer_vad = _buffer_vad[512:]
                    prob_voz = _vad_silero.process(chunk_vad)

                    if prob_voz > _umbral_voz:
                        if not _en_intervencion:
                            _en_intervencion = True
                            if not _turno_en_validacion:
                                iniciar_turno_provisional()

                        _muestras_silencio = 0

                        if _turno_en_validacion:
                            _buffer_idioma = np.concatenate([_buffer_idioma, chunk_vad])
                            max_samples = int(16000 * VL_BUFFER_SEG)

                            _buffer_idioma = (_buffer_idioma[-max_samples:])

                    else:
                        _muestras_silencio += 512
                        if _muestras_silencio >= int(16000 * SILENCIO_FIN_TURNO_SEG):
                            _en_intervencion = False

                if _turno_en_validacion:
                    guardar_audio_recuperacion(data)
                    ahora = time.monotonic()

                    if (
                        len(_buffer_idioma) >= int(16000 * VL_MIN_AUDIO_SEG) and ahora - _ultimo_intento_clasificar >= VL_INTERVALO_SEG and not _clasificando_idioma):
                        _ultimo_intento_clasificar = ahora
                        _clasificando_idioma = True
                        _executor_voxlingua.submit(_tarea_clasificar_idioma, _buffer_idioma.copy(), _generacion_clasificador)

            procesar_bloque_vosk(rec, data)

            if _turno_en_validacion:
                if _bytes_recuperacion >= int(SR_ACTUAL * 2* (BUFFER_RECUPERACION_SEG - 0.2)):

                    log.info("Timeout de clasificación. " "Conservando idioma actual.")

                    confirmar_turno_provisional()

    except Exception as e:
        log.exception(
            "Error en procesar_audio: %s", e
        )

    finally:
        ventana.after(
            TICK_PROCESAR_MS, procesar_audio
        )

        
def _procesar_cola_ui():
    try:
        for _ in range(20):
            try:
                msg = cola_ui.get_nowait()
            except queue.Empty:
                break
            tipo = msg[0]
            if tipo == "aplicar_resultado":
                _, entrada, texto_puntuado, trad = msg
                try:
                    _aplicar_resultado(entrada, texto_puntuado, trad)
                except Exception as e:
                    log.exception("Error aplicando resultado: %s", e)

            elif tipo == "resultado_clasificacion":
                _, idioma, confianza, generacion = msg
                if (generacion == _generacion_clasificador and _config_visual.get("deteccion_automatica_idioma", True)):
                    gestionar_resultado_clasificacion(idioma, confianza)
            elif tipo == "cambiar_idioma_manual":
                cambiar_idioma(None)
            else:
                log.warning("Mensaje UI desconocido: %r", tipo)

    except Exception as e:
        log.exception(
            "Error en _procesar_cola_ui: %s", e
        )

    finally:
        ventana.after(50, _procesar_cola_ui)


def recibir_texto_vosk(contenido):
    if (_config_visual.get("deteccion_automatica_idioma", True) and _turno_en_validacion):
        _textos_provisionales.append(contenido)
    else:
        _confirmar_con_buffer(contenido)


def vigilar_stream():
    try:
        if stream_actual["stream"] is not None:
            delta = time.time() - stream_actual["ultimo_bloque"]
            if delta > TIMEOUT_STREAM_SEG:
                dev_actual = next((d for d in _dispositivos_cache if d["indice"] == stream_actual["device"]), None)
                if dev_actual:
                    abrir_stream(dev_actual)
    except Exception as e:
        log.exception("Error en vigilar_stream: %s", e)
    finally:
        ventana.after(TICK_WATCHDOG_MS, vigilar_stream)


# ---------------------------------------------------------------------------
# Carga en segundo plano
# ---------------------------------------------------------------------------
def actualizar_mensaje(texto_nuevo):
    ventana.after(0, lambda: mensaje_carga.config(text=texto_nuevo))


def _cargar_modelos_bg():
    for idioma, ruta in RUTAS_MODELOS.items():
        if _modelos[idioma] is None:
            actualizar_mensaje(f"Ami está buscando su diccionario en {idioma.lower()}...")
            _modelos[idioma] = Model(ruta)


def _cargar_traduccion_bg():
    if os.path.isdir(CARPETA_TRADUCCIONES):
        archivos_locales = [f for f in os.listdir(CARPETA_TRADUCCIONES) if f.endswith(".argosmodel")]
        for archivo in archivos_locales:
            ruta = os.path.join(CARPETA_TRADUCCIONES, archivo)
            actualizar_mensaje(f"Ami está preparando {archivo}...")
            try:
                argostranslate.package.install_from_path(ruta)
            except Exception:
                pass
    installed_languages = argostranslate.translate.get_installed_languages()
    idiomas = {lang.code: lang for lang in installed_languages}

    for from_code, to_code in [("en", "es"), ("es", "en")]:
        if from_code in idiomas and to_code in idiomas:
            try:
                _traducciones[(from_code, to_code)] = idiomas[from_code].get_translation(idiomas[to_code])
            except Exception:
                pass

    for (from_code, to_code), traductor in _traducciones.items():
        try:
            actualizar_mensaje(f"Ami está practicando {from_code}→{to_code}...")
            traductor.translate("hello" if from_code == "en" else "hola")
        except Exception:
            pass


def _cargar_puntuador_bg():
    global _puntuador
    try:
        actualizar_mensaje("Ami está puliendo su gramática...")
        _puntuador = PunctCapSegModelONNX.from_pretrained("pcs_47lang")
    except Exception as e:
        log.error("No se pudo cargar el puntuador: %s", e)
        _puntuador = None


def proceso_de_carga():
    try:
        _cargar_modelos_bg()
        _cargar_traduccion_bg()
        _cargar_voxlingua()
        _cargar_puntuador_bg()
        _cargar_vad()
        ventana.after(0, construir_interfaz_principal)
    except Exception as e:
        log.exception("Error fatal durante la carga: %s", e)
        ventana.after(0, lambda: mensaje_carga.config(text="Error al cargar. Revisa la consola."))


# ---------------------------------------------------------------------------
# Apariencia
# ---------------------------------------------------------------------------
def _colores_boton_tema():
    bg = _config_visual.get("color_boton", "#2A2A2A")
    fg = _config_visual.get("color_boton_fg", "white")
    active_bg = _config_visual.get("color_boton_active", "#3A3A3A")
    active_fg = _config_visual.get("color_boton_active_fg", fg)
    return bg, fg, active_bg, active_fg


def _recorrer_widgets(widget):
    yield widget
    for hijo in widget.winfo_children():
        yield from _recorrer_widgets(hijo)


def _aplicar_colores_botones(raiz):
    bg, fg, active_bg, active_fg = _colores_boton_tema()
    fondo_ventana = _config_visual.get("fondo_ventana", "#000000")

    try:
        for w in _recorrer_widgets(raiz):
            if not w.winfo_exists():
                continue
            if getattr(w, "_color_fijo", False):
                continue
            if isinstance(w, tk.Button):
                try:
                    w.configure(bg=bg,
                                fg=fg,
                                activebackground=active_bg,
                                activeforeground=active_fg)
                except Exception:
                    pass
            elif isinstance(w, tk.Checkbutton):
                try:
                    w.configure(bg=fondo_ventana,
                                fg=fg,
                                activebackground=fondo_ventana,
                                activeforeground=active_fg,
                                selectcolor=bg)
                except Exception:
                    pass
    except Exception as e:
        log.warning("Error aplicando colores a botones: %s", e)


def _registrar_fuente_ui(widget,
                         tamaño_offset=0,
                         estilo="normal"):

    _widgets_fuente_ui.append({
        "widget": widget,
        "tamaño_offset": tamaño_offset,
        "estilo": estilo,
    })
    _aplicar_fuente_ui_widget(widget,
                              tamaño_offset,
                              estilo)
    return widget


def _aplicar_fuente_ui_widget(widget,
                              tamaño_offset,
                              estilo):
    try:
        if not widget.winfo_exists():
            return
        base = _config_visual.get("tamaño_ui", 11)
        tamaño_final = max(6, base + tamaño_offset)
        widget.configure(font=(_config_visual["fuente_ui"],
                               tamaño_final,
                               estilo))
    except Exception:
        pass


def _aplicar_fuente_ui_a_todos():
    vivos = []
    for entrada in _widgets_fuente_ui:
        w = entrada["widget"]
        try:
            if not w.winfo_exists():
                continue
            _aplicar_fuente_ui_widget(w,
                                      entrada["tamaño_offset"],
                                      entrada["estilo"])
            vivos.append(entrada)
        except Exception:
            continue
    _widgets_fuente_ui[:] = vivos


def _label_ui(parent, texto,
              tamaño_offset=0,
              estilo="normal",
              **kwargs):
    lbl = tk.Label(parent,
                   text=texto,
                   **kwargs)
    _registrar_fuente_ui(lbl,
                         tamaño_offset=tamaño_offset,
                         estilo=estilo)
    return lbl


def actualizar_preview_widget():
    global preview_widget, _config_visual
    if preview_widget is None or not preview_widget.winfo_exists():
        return
    try:
        if not preview_widget.winfo_exists():
            return
    except Exception:
        return

    if _config_visual["modo_vision_cansada"]:
        fondo_txt = "#2A2A2A"
        color_txt = "#E0E0E0"
    else:
        fondo_txt = _config_visual["fondo_texto"]
        color_txt = "white"

    preview_widget.configure(bg=fondo_txt)

    fuente = _config_visual["fuente_dislexia"] if _config_visual["modo_dislexia"] else _config_visual["fuente_texto"]
    tamaño = _config_visual["tamaño_texto"]

    estilo = "bold" if _config_visual["modo_tdah"] else "normal"
    if _config_visual["modo_tdah"]:
        preview_widget.configure(spacing1=6, spacing2=6, spacing3=12)
    else:
        preview_widget.configure(spacing1=0, spacing2=0, spacing3=0)

    col_ingles = _config_visual["color_ingles"]
    col_español = _config_visual["color_español"]
    col_trad = _config_visual.get("color_traduccion", "#999999")
    col_parcial = _config_visual.get("color_parcial", "#B8B8B8")

    if _config_visual["modo_daltonismo_rv"]:
        col_ingles = "#0072B2"
        col_español = "#E69F00"
    elif _config_visual["modo_daltonismo_aa"]:
        col_ingles = "#CC79A7"
        col_español = "#009E73"
    elif _config_visual["modo_vision_cansada"]:
        col_ingles = "#8ab4f8"
        col_español = "#f9ab76"
        col_trad = _config_visual.get("color_traduccion", "#999999")
        col_parcial = "#AAAAAA"

    preview_widget.configure(font=(fuente, tamaño, estilo))

    preview_widget.config(state="normal")
    preview_widget.delete("1.0", "end")
    preview_widget.configure(fg=color_txt)

    preview_widget.tag_configure("ingles",
                                 font=(fuente, tamaño, estilo),
                                 foreground=col_ingles)
    preview_widget.tag_configure("español",
                                 font=(fuente, tamaño, estilo),
                                 foreground=col_español)
    estilo_trad = "normal" if _config_visual["modo_tdah"] else "italic"
    preview_widget.tag_configure("traduccion",
                                 font=(fuente, tamaño, estilo_trad),
                                 foreground=col_trad)

    preview_widget.insert("end", "Rep: ", "ingles")
    preview_widget.insert("end", "Hello, how are you?\n", "ingles")
    preview_widget.insert("end", "Lep: ", "español")
    preview_widget.insert("end", "Hola, ¿cómo estás?\n", "español")
    preview_widget.insert("end", "     ES: ", "traduccion")
    preview_widget.insert("end", "Hello, how are you?\n", "traduccion")

    preview_widget.config(state="disabled")


def forzar_redibujado_texto():
    if texto is None:
        return
        
    texto.config(state="normal")
    texto.delete("1.0", "end")
    
    for entrada in historial:
        texto.insert("end", entrada["original"] + "\n", entrada["etiqueta"])
        if entrada.get("traduccion"):
            texto.insert("end", entrada["traduccion"] + "\n", "traduccion")
        texto.insert("end", "\n")
        
    texto.insert("end", "\n")
    texto.mark_set("parcial_inicio", "end-1c")
    texto.mark_gravity("parcial_inicio", tk.LEFT)
    texto.config(state="disabled")
    texto.see("end")

    actualizar_texto_parcial()


def aplicar_tema_visual():
    global _config_visual

    _aplicar_fuente_ui_a_todos()
    actualizar_preview_widget()
    
    if texto is None:
        return 

    ventana.configure(bg=_config_visual["fondo_ventana"])
    if frame_fuente_ref:
        frame_fuente_ref.configure(bg=_config_visual["fondo_ventana"])
    if frame_toggles_ref:
        frame_toggles_ref.configure(bg=_config_visual["fondo_ventana"])
    if frame_panel_superior:
        frame_panel_superior.configure(bg=_config_visual["fondo_ventana"])

    if _config_visual["modo_vision_cansada"]:
        fondo_txt = "#2A2A2A"
    else:
        fondo_txt = _config_visual["fondo_texto"]
    
    texto.configure(bg=fondo_txt)

    fuente_texto_actual = _config_visual["fuente_texto"]
    if _config_visual["modo_dislexia"]:
        fuente_texto_actual = _config_visual["fuente_dislexia"]
    tamaño = _config_visual["tamaño_texto"]

    if _config_visual["modo_tdah"]:
        estilo_tdah = "bold"
        texto.configure(spacing1=6,
                        spacing2=6,
                        spacing3=12)
    else:
        estilo_tdah = "normal"
        texto.configure(spacing1=0,
                        spacing2=0,
                        spacing3=0)

    col_ingles = _config_visual.get("color_ingles", "#a5d7f2")
    col_español = _config_visual.get("color_español", "#f2b27a")
    col_trad = _config_visual.get("color_traduccion", "#999999")
    col_parcial = _config_visual.get("color_parcial", "#B8B8B8")

    if _config_visual["modo_daltonismo_rv"]:
        col_ingles = "#0072B2"
        col_español = "#E69F00"
    elif _config_visual["modo_daltonismo_aa"]:
        col_ingles = "#CC79A7"
        col_español = "#009E73"
    elif _config_visual["modo_vision_cansada"]:
        col_ingles = "#8ab4f8"
        col_español = "#f9ab76"
        col_trad = "#CCCCCC"
        col_parcial = "#AAAAAA"

    texto.tag_configure("ingles",
                        font=(fuente_texto_actual, tamaño, estilo_tdah),
                        foreground=col_ingles)
    texto.tag_configure("español",
                        font=(fuente_texto_actual, tamaño, estilo_tdah),
                        foreground=col_español)
    
    estilo_trad = "normal" if _config_visual["modo_tdah"] else "italic"
    texto.tag_configure("traduccion",
                        font=(fuente_texto_actual, tamaño, estilo_trad),
                        foreground=col_trad)
    texto.tag_configure("parcial",
                        font=(fuente_texto_actual, tamaño, "italic"),
                        foreground=col_parcial)
    
    texto.configure(font=(fuente_texto_actual, tamaño))

    _aplicar_colores_botones(ventana)
    if ventana_ajustes_ref is not None and ventana_ajustes_ref.winfo_exists():
        _aplicar_colores_botones(ventana_ajustes_ref)

    forzar_redibujado_texto()
    actualizar_preview_widget()
    guardar_configuracion()


def aplicar_fondo():
    global label_fondo, _imagen_fondo_original, _config_visual, _ultimo_tamaño_fondo
    ruta = _config_visual.get("imagen_fondo")
    if not ruta or not os.path.exists(ruta):
        if label_fondo:
            label_fondo.destroy()
            label_fondo = None
        _ultimo_tamaño_fondo = (0, 0)
        return
    try:
        from PIL import Image, ImageTk, ImageEnhance

        w_ventana = ventana.winfo_width()
        h_ventana = ventana.winfo_height()
        if w_ventana < 100:
            w_ventana = 900
        if h_ventana < 100:
            h_ventana = 650

        tamaño_actual = (w_ventana, h_ventana)
        if (tamaño_actual == _ultimo_tamaño_fondo
                and _imagen_fondo_original is not None
                and getattr(_imagen_fondo_original, "filename", "") == ruta):
            return
        _ultimo_tamaño_fondo = tamaño_actual

        if (_imagen_fondo_original is None
                or getattr(_imagen_fondo_original, "filename", "") != ruta):
            _imagen_fondo_original = Image.open(ruta)
            _imagen_fondo_original.filename = ruta

        img = _imagen_fondo_original.copy()
        ratio_img = img.width / img.height
        ratio_ven = w_ventana / h_ventana

        if ratio_ven > ratio_img:
            nuevo_w = w_ventana
            nuevo_h = int(w_ventana / ratio_img)
        else:
            nuevo_h = h_ventana
            nuevo_w = int(h_ventana * ratio_img)

        img = img.resize((nuevo_w, nuevo_h), Image.Resampling.LANCZOS)
        left = (img.width - w_ventana) / 2
        top = (img.height - h_ventana) / 2
        right = (img.width + w_ventana) / 2
        bottom = (img.height + h_ventana) / 2
        img = img.crop((left, top, right, bottom))
        img = ImageEnhance.Brightness(img).enhance(0.35)
        foto = ImageTk.PhotoImage(img)

        if label_fondo is None:
            label_fondo = tk.Label(ventana,
                                   image=foto,
                                   bd=0)
            label_fondo.image = foto
            label_fondo.place(x=0,
                              y=0,
                              relwidth=1,
                              relheight=1)
            label_fondo.lower()
        else:
            label_fondo.config(image=foto)
            label_fondo.image = foto

    except Exception as e:
        log.error("Error al cargar la imagen de fondo: %s", e)


def _on_resize_ventana(event=None):
    global _job_fondo
    if _job_fondo is not None:
        try:
            ventana.after_cancel(_job_fondo)
        except Exception:
            pass
    _job_fondo = ventana.after(150, aplicar_fondo)


def aplicar_tema_predefinido(nombre_tema):
    global _config_visual
    if nombre_tema in TEMAS_PREDEFINIDOS:
        tema = TEMAS_PREDEFINIDOS[nombre_tema]
        _config_visual.update(tema)
        guardar_configuracion()
        aplicar_tema_visual()



# ---------------------------------------------------------------------------
# Ventana de Ajustes
# ---------------------------------------------------------------------------
def abrir_ventana_ajustes():
    global _config_visual, preview_widget, ventana_ajustes_ref

    _widgets_fuente_ui[:] = [
        e for e in _widgets_fuente_ui
        if e["widget"].winfo_exists()
    ]

    ventana_ajustes = tk.Toplevel(ventana)
    ventana_ajustes_ref = ventana_ajustes
    ventana_ajustes.title("Ajustes de Terpi")
    ventana_ajustes.geometry("500x650")
    ventana_ajustes.configure(bg="#121212")
    ventana_ajustes.transient(ventana)
    ventana_ajustes.grab_set()

    frame_bottom = tk.Frame(ventana_ajustes,
                            bg="#121212")
    frame_bottom.pack(side="bottom",
                      fill="x",
                      padx=10,
                      pady=10)

    frame_preview = tk.Frame(frame_bottom,
                             bg="#1E1E1E",
                             bd=1,
                             relief="solid")
    frame_preview.pack(fill="x",
                       pady=(0, 10))

    preview_widget = tk.Text(frame_preview,
                             height=4, wrap="word",
                             bg="#1E1E1E",
                             fg="white",
                             padx=10, pady=10,
                             state="disabled")
    preview_widget.pack(fill="both", expand=True)
    ventana_ajustes.after(50, actualizar_preview_widget)

    actualizar_preview_widget()

    notebook = ttk.Notebook(ventana_ajustes)
    notebook.pack(fill="both",
                  expand=True,
                  padx=10,
                  pady=10)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TNotebook",
                    background="#121212",
                    borderwidth=0)
    style.configure("TNotebook.Tab",
                    background="#2A2A2A",
                    foreground="white",
                    padding=[10, 5])
    style.map("TNotebook.Tab",
              background=[("selected", "#121212")],
              foreground=[("selected", "#a5d7f2")])

    guardar_configuracion()

    # -----------------------------------------------------------------------
    # PESTAÑA 1: Apariencia
    # -----------------------------------------------------------------------
    tab_apariencia = tk.Frame(notebook,
                              bg="#121212")
    notebook.add(tab_apariencia,
                 text="🎨 Apariencia")

    contenedor_apariencia = tk.Frame(tab_apariencia,
                                     bg="#121212")
    contenedor_apariencia.pack(fill="both",
                               expand=True,
                               padx=10,
                               pady=10)

    col_izq = tk.Frame(contenedor_apariencia, bg="#121212")
    col_izq.pack(side="left",
                 fill="both",
                 expand=True,
                 padx=(0, 10))

    col_der = tk.Frame(contenedor_apariencia, bg="#121212")
    col_der.pack(side="left",
                 fill="both",
                 expand=True,
                 padx=(10, 0))

        # ===== COLUMNA IZQUIERDA =====
    
    _label_ui(col_izq, "Temas Preestablecidos",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    combo_temas = ttk.Combobox(col_izq,
                               values=list(TEMAS_PREDEFINIDOS.keys()),
                               state="readonly")
    combo_temas.pack(pady=5)
    combo_temas.set("Terpi Original")

    def cambiar_tema(event):
        aplicar_tema_predefinido(combo_temas.get())

    combo_temas.bind("<<ComboboxSelected>>", cambiar_tema)

    ttk.Separator(col_izq, orient="horizontal").pack(fill="x", padx=20, pady=15)

        # --- Color de fondo (Controles) ---
    _label_ui(col_izq,
              "Color de fondo (Controles)",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    def elegir_color_fondo():
        color = colorchooser.askcolor(
            title="Elige el color de fondo",
            initialcolor=_config_visual["fondo_ventana"])[1]
        if color:
            _config_visual["fondo_ventana"] = color
            aplicar_tema_visual()

    btn_c_fondo = tk.Button(col_izq,
                            text="Elegir Color de Ventana",
                            command=elegir_color_fondo,
                            bg="#2A2A2A",
                            fg="white",
                            relief="flat")
    _registrar_fuente_ui(btn_c_fondo, tamaño_offset=-1)
    btn_c_fondo.pack(pady=5)

        # --- Color de fondo (Área de texto) ---
    _label_ui(col_izq,
              "Color de fondo (Área de texto)",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    def elegir_color_texto():
        color = colorchooser.askcolor(
            title="Elige el color del área de texto",
            initialcolor=_config_visual["fondo_texto"])[1]
        if color:
            _config_visual["fondo_texto"] = color

            aplicar_tema_visual()

    btn_c_texto = tk.Button(col_izq,
                            text="Elegir Color del Área de Texto",
                            command=elegir_color_texto,
                            bg="#2A2A2A",
                            fg="white",
                            relief="flat")
    _registrar_fuente_ui(btn_c_texto, tamaño_offset=-1)
    btn_c_texto.pack(pady=5)

    ttk.Separator(col_izq,
                  orient="horizontal").pack(fill="x",
                                            padx=20,
                                            pady=15)

        # --- Colores de los botones ---
    _label_ui(col_izq, "Colores de los Botones",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    def elegir_color_boton():
        color = colorchooser.askcolor(
            title="Elige el color de fondo de los botones",
            initialcolor=_config_visual.get("color_boton", "#2A2A2A"))[1]
        if color:
            _config_visual["color_boton"] = color
            aplicar_tema_visual()

    btn_cb_fondo = tk.Button(col_izq,
                             text="Color de Fondo",
                             command=elegir_color_boton,
                             bg="#2A2A2A",
                             fg="white",
                             relief="flat")
    _registrar_fuente_ui(btn_cb_fondo,
                         tamaño_offset=-1)
    btn_cb_fondo._color_fijo = True
    btn_cb_fondo.pack(pady=3)

    def elegir_color_boton_fg():
        color = colorchooser.askcolor(
            title="Elige el color del texto de los botones",
            initialcolor=_config_visual.get("color_boton_fg", "white"))[1]
        if color:
            _config_visual["color_boton_fg"] = color
            aplicar_tema_visual()

    btn_cb_fg = tk.Button(col_izq,
                          text="Color del Texto",
                          command=elegir_color_boton_fg,
                          bg="#2A2A2A",
                          fg="white",
                          relief="flat")
    _registrar_fuente_ui(btn_cb_fg, tamaño_offset=-1)
    btn_cb_fg._color_fijo = True
    btn_cb_fg.pack(pady=3)

    def elegir_color_boton_active():
        color = colorchooser.askcolor(
            title="Elige el color del boton al hacer click",
            initialcolor=_config_visual.get("color_boton_active", "#3A3A3A"))[1]
        if color:
            _config_visual["color_boton_active"] = color
            aplicar_tema_visual()

    btn_cb_active = tk.Button(col_izq,
                              text="Color del Click",
                              command=elegir_color_boton_active,
                              bg="#2A2A2A",
                              fg="white",
                              relief="flat")
    _registrar_fuente_ui(btn_cb_active, tamaño_offset=-1)
    btn_cb_active._color_fijo = True
    btn_cb_active.pack(pady=3)

        # ===== COLUMNA DERECHA =====

#TODO: cargar la imágen en el area de texto y no en el fondo
## Reservar para texturas (?)
    _label_ui(col_der, "Imagen de Fondo",
              tamaño_offset=0, estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))


    def elegir_imagen_fondo():
        ruta = filedialog.askopenfilename(
            title="Elige una imagen de fondo",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg")])
        if ruta:
            _config_visual["imagen_fondo"] = ruta
            aplicar_fondo()
            guardar_configuracion()

    btn_img_set = tk.Button(col_der,
                            text="Elegir Imagen",
                            command=elegir_imagen_fondo,
                            bg="#2A2A2A",
                            fg="white",
                            relief="flat")
    _registrar_fuente_ui(btn_img_set, tamaño_offset=-1)
    btn_img_set.pack(pady=3)

    btn_img_quit = tk.Button(col_der,
                             text="Quitar Imagen",
                             command=lambda: [_config_visual.update({"imagen_fondo": None}),
                                 aplicar_fondo(),
                                 guardar_configuracion()
                             ],
                             bg="#2A2A2A",
                             fg="white",
                             relief="flat")
    _registrar_fuente_ui(btn_img_quit, tamaño_offset=-1)
    btn_img_quit.pack(pady=3)

    actualizar_preview_widget()

    # -----------------------------------------------------------------------
    # PESTAÑA 2: Tipografía (slider arriba, dos columnas abajo)
    # -----------------------------------------------------------------------
    tab_tipografia = tk.Frame(notebook,
                              bg="#121212")
    notebook.add(tab_tipografia,
                 text="🔤 Tipografía")

    contenedor_tipo = tk.Frame(tab_tipografia,
                               bg="#121212")
    contenedor_tipo.pack(fill="both",
                         expand=True,
                         padx=10,
                         pady=(20, 10))

        # --- Fila superior: slider de tamaño del texto ---
    
    _label_ui(contenedor_tipo,
              "Tamaño del Texto",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(5, 5))

    var_tamaño_texto = tk.IntVar(value=_config_visual["tamaño_texto"])

    def cambiar_tamaño(event=None):
        try:
            _config_visual["tamaño_texto"] = var_tamaño_texto.get()
            aplicar_tema_visual()
        except Exception:
            pass

    tk.Scale(contenedor_tipo,
             from_=8, to=32,
             orient="horizontal",
             variable=var_tamaño_texto,
             bg="#121212",
             fg="white",
             highlightthickness=0,
             troughcolor="#2A2A2A",
             activebackground="#a5d7f2",
             command=cambiar_tamaño,
             length=400).pack(fill="x",
                              padx=40,
                              pady=5)

    ttk.Separator(contenedor_tipo, orient="horizontal").pack(fill="x",
                                                             padx=20,
                                                             pady=(15, 10))

        # --- Fila inferior  ---
    frame_superior = tk.Frame(contenedor_tipo, bg="#121212")
    frame_superior.pack(fill="both",
                        expand=True)

    col_tipo_izq = tk.Frame(frame_superior,
                            bg="#121212")
    col_tipo_izq.pack(side="left",
                      fill="both",
                      expand=True,
                      padx=(0, 10))

    ttk.Separator(frame_superior, orient="vertical").pack(side="left",
                                                          fill="y",
                                                          padx=5)

    col_tipo_der = tk.Frame(frame_superior, bg="#121212")
    col_tipo_der.pack(side="left",
                      fill="both",
                      expand=True,
                      padx=(10, 0))

    todas_las_fuentes = sorted(list(tkfont.families()))

        # ===== COLUMNA IZQUIERDA: Fuentes =====

    _label_ui(col_tipo_izq,
              "Fuente de la Interfaz",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    combo_ui = ttk.Combobox(col_tipo_izq,
                            values=todas_las_fuentes,
                            state="readonly")
    combo_ui.set(_config_visual["fuente_ui"])
    combo_ui.pack()
    combo_ui.bind("<<ComboboxSelected>>",
                  lambda e: [
                      _config_visual.update({"fuente_ui": combo_ui.get()}),
                      aplicar_tema_visual()
                  ])

    _label_ui(col_tipo_izq, "Fuente de la Transcripción",
              tamaño_offset=0, estilo="bold",
              bg="#121212", fg="white").pack(pady=(15, 5))

    combo_texto = ttk.Combobox(col_tipo_izq,
                               values=todas_las_fuentes,
                               state="readonly")
    combo_texto.set(_config_visual["fuente_texto"])
    combo_texto.pack()
    combo_texto.bind("<<ComboboxSelected>>",
                     lambda e: [
                         _config_visual.update({"fuente_texto": combo_texto.get()}),
                         aplicar_tema_visual()
                     ])

        # ===== COLUMNA DERECHA: Colores =====

    _label_ui(col_tipo_der,
              "Color de la Transcripción (Inglés)",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    def elegir_color_ingles():
        color = colorchooser.askcolor(
            title="Elige el color para el Inglés",
            initialcolor=_config_visual["color_ingles"])[1]
        if color:
            _config_visual["color_ingles"] = color
            aplicar_tema_visual()

    btn_ci = tk.Button(col_tipo_der,
                       text="Elegir Color Inglés",
                       command=elegir_color_ingles,
                       bg="#2A2A2A",
                       fg="white",
                       relief="flat")
    _registrar_fuente_ui(btn_ci,
                         tamaño_offset=-1)
    btn_ci.pack(pady=5)

    _label_ui(col_tipo_der, "Color de la Transcripción (Español)",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    def elegir_color_español():
        color = colorchooser.askcolor(
            title="Elige el color para el Español",
            initialcolor=_config_visual["color_español"])[1]
        if color:
            _config_visual["color_español"] = color
            aplicar_tema_visual()

    btn_ce = tk.Button(col_tipo_der,
                       text="Elegir Color Español",
                       command=elegir_color_español,
                       bg="#2A2A2A",
                       fg="white",
                       relief="flat")
    _registrar_fuente_ui(btn_ce, tamaño_offset=-1)
    btn_ce.pack(pady=5)

    _label_ui(col_tipo_der,
              "Color de la Traducción",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(15, 5))

    def elegir_color_traduccion():
        color = colorchooser.askcolor(
            title="Elige el color para la Traducción",
            initialcolor=_config_visual.get("color_traduccion", "#999999"))[1]
        if color:
            _config_visual["color_traduccion"] = color
            aplicar_tema_visual()

    btn_ct = tk.Button(col_tipo_der,
                       text="Elegir Color Traducción",
                       command=elegir_color_traduccion,
                       bg="#2A2A2A",
                       fg="white",
                       relief="flat")
    _registrar_fuente_ui(btn_ct, tamaño_offset=-1)
    btn_ct.pack(pady=5)

    actualizar_preview_widget()

    
    # -----------------------------------------------------------------------
    # PESTAÑA 3: Accesibilidad
    # -----------------------------------------------------------------------
    tab_accesibilidad = tk.Frame(notebook, bg="#121212")
    notebook.add(tab_accesibilidad, text="♿ Accesibilidad")

        # --- Modo dislexia ---
    var_dislexia = tk.BooleanVar(value=_config_visual["modo_dislexia"])
    fuentes_dislexia = ["OpenDyslexic", "Atkinson Hyperlegible",
                        "Comic Sans MS", "Verdana"]
    combo_dislexia = ttk.Combobox(
        tab_accesibilidad,
        values=fuentes_dislexia,
        state="readonly" if _config_visual["modo_dislexia"] else "disabled")
    combo_dislexia.set(_config_visual["fuente_dislexia"])

    def toggle_dislexia():
        _config_visual["modo_dislexia"] = var_dislexia.get()
        combo_dislexia.config(
            state="readonly" if var_dislexia.get() else "disabled")
        aplicar_tema_visual()

    chk_dislexia = tk.Checkbutton(tab_accesibilidad,
                                  text="Activar Modo Dislexia",
                                  variable=var_dislexia,
                                  command=toggle_dislexia,
                                  bg="#121212",
                                  fg="white",
                                  selectcolor="#2A2A2A",
                                  activebackground="#121212",
                                  activeforeground="white")
    _registrar_fuente_ui(chk_dislexia, tamaño_offset=-1)
    chk_dislexia.pack(pady=(15, 2))

    combo_dislexia.pack(pady=5)
    combo_dislexia.bind("<<ComboboxSelected>>",
                        lambda e: [
                            _config_visual.update(
                                {"fuente_dislexia": combo_dislexia.get()}),
                            aplicar_tema_visual()
                        ])

        # --- Modo TDAH ---
    var_tdah = tk.BooleanVar(value=_config_visual["modo_tdah"])
    chk_tdah = tk.Checkbutton(tab_accesibilidad,
                              text="Modo TDAH (Enfoque)",
                              variable=var_tdah,
                              command=lambda: [
                                  _config_visual.update({"modo_tdah": var_tdah.get()}),
                                  aplicar_tema_visual()
                              ],
                              bg="#121212",
                              fg="white",
                              selectcolor="#2A2A2A",
                              activebackground="#121212",
                              activeforeground="white")
    _registrar_fuente_ui(chk_tdah, tamaño_offset=-1)
    chk_tdah.pack(pady=2)

        # --- Visión cansada ---
    var_cansada = tk.BooleanVar(value=_config_visual["modo_vision_cansada"])
    chk_cansada = tk.Checkbutton(tab_accesibilidad,
                                 text="Modo Visión Cansada (Bajo Contraste)",
                                 variable=var_cansada,
                                 command=lambda: [
                                     _config_visual.update(
                                         {"modo_vision_cansada": var_cansada.get()}),
                                     aplicar_tema_visual()
                                 ],
                                 bg="#121212",
                                 fg="white",
                                 selectcolor="#2A2A2A",
                                 activebackground="#121212",
                                 activeforeground="white")
    _registrar_fuente_ui(chk_cansada, tamaño_offset=-1)
    chk_cansada.pack(pady=2)

        # --- Daltonismo rojo-verde ---
    var_daltonismo_rv = tk.BooleanVar(value=_config_visual["modo_daltonismo_rv"])
    chk_dalt_rv = tk.Checkbutton(tab_accesibilidad,
                                 text="Daltonismo Rojo-Verde",
                                 variable=var_daltonismo_rv,
                                 command=lambda: [
                                     _config_visual.update(
                                         {"modo_daltonismo_rv": var_daltonismo_rv.get()}),
                                     aplicar_tema_visual()
                                 ],
                                 bg="#121212",
                                 fg="white",
                                 selectcolor="#2A2A2A",
                                 activebackground="#121212",
                                 activeforeground="white")
    _registrar_fuente_ui(chk_dalt_rv, tamaño_offset=-1)
    chk_dalt_rv.pack(pady=2)

        # --- Daltonismo amarillo-azul ---
    var_daltonismo_aa = tk.BooleanVar(value=_config_visual["modo_daltonismo_aa"])
    chk_dalt_aa = tk.Checkbutton(tab_accesibilidad,
                                 text="Daltonismo Amarillo-Azul",
                                 variable=var_daltonismo_aa,
                                 command=lambda: [
                                     _config_visual.update(
                                         {"modo_daltonismo_aa": var_daltonismo_aa.get()}),
                                     aplicar_tema_visual()
                                 ],
                                 bg="#121212",
                                 fg="white",
                                 selectcolor="#2A2A2A",
                                 activebackground="#121212",
                                 activeforeground="white")
    _registrar_fuente_ui(chk_dalt_aa, tamaño_offset=-1)
    chk_dalt_aa.pack(pady=2)

    var_deteccion_idioma = tk.BooleanVar(value=_config_visual.get("deteccion_automatica_idioma", True))
    chk_deteccion = tk.Checkbutton(tab_accesibilidad,
                                   text="Detección automática de idioma",
                                   variable=var_deteccion_idioma,
                                   command=lambda: [
                                       _config_visual.update({"deteccion_automatica_idioma": var_deteccion_idioma.get()}),
                                       guardar_configuracion()
                                   ],
                                   bg="#121212",
                                   fg="white",
                                   selectcolor="#2A2A2A",
                                   activebackground="#121212",
                                   activeforeground="white")
    _registrar_fuente_ui(chk_deteccion, tamaño_offset=-1)
    chk_deteccion.pack(pady=2)

    actualizar_preview_widget()
    guardar_configuracion()

    _aplicar_colores_botones(ventana_ajustes)

    # -----------------------------------------------------------------------
    # PESTAÑA 4: Acerca de
    # -----------------------------------------------------------------------

    tab_acerca = tk.Frame(notebook, bg="#121212")
    notebook.add(tab_acerca, text="ℹ️ Acerca de")

    _label_ui(tab_acerca, "Terpi",
              tamaño_offset=+7,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(40, 0))
    _label_ui(tab_acerca,
              f"Versión {VERSION_TERPI}",
              tamaño_offset=0,
              bg="#121212",
              fg="#BBBBBB").pack(pady=(0, 3))
    _label_ui(tab_acerca,
              f"Licencia {LICENCIA_TERPI}",
              tamaño_offset=0,
              bg="#121212",
              fg="#BBBBBB").pack(pady=(0, 20))

    ttk.Separator(tab_acerca,
                  orient="horizontal").pack(fill="x",
                                            padx=40,
                                            pady=15)

    _label_ui(tab_acerca,
              "Comentarios, sugerencias y reportes de bugs:",
              tamaño_offset=0,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(10, 5))

    def _abrir_url_ajustes(url):
        import webbrowser
        webbrowser.open(url)

    frame_soporte = tk.Frame(tab_acerca,
                             bg="#121212")
    frame_soporte.pack(pady=5)

    btn_github = tk.Button(frame_soporte,
                           text="GitHub Issues",
                           command=lambda: _abrir_url_ajustes(URL_GITHUB_ISSUES),
                           bg="#2A2A2A",
                           fg="white",
                           activebackground="#3A3A3A",
                           activeforeground="white",
                           relief="flat",
                           padx=12,
                           pady=5)
    _registrar_fuente_ui(btn_github,
                         tamaño_offset=-1)
    btn_github.pack(side="left",
                    padx=3)

    btn_facebook = tk.Button(frame_soporte,
                             text="Facebook",
                             command=lambda: _abrir_url_ajustes(URL_FACEBOOK),
                             bg="#2A2A2A",
                             fg="white",
                             activebackground="#3A3A3A",
                             activeforeground="white",
                             relief="flat",
                             padx=12,
                             pady=5)
    _registrar_fuente_ui(btn_facebook,
                         tamaño_offset=-1)
    btn_facebook.pack(side="left",
                      padx=3)

    btn_email = tk.Button(frame_soporte,
                          text="Correo electronico",
                          command=lambda: _abrir_url_ajustes(f"mailto:{EMAIL_CONTACTO}?subject=Terpi%20v{VERSION_TERPI}"),
                          bg="#2A2A2A",
                          fg="white",
                          activebackground="#3A3A3A",
                          activeforeground="white",
                          relief="flat",
                          padx=12,
                          pady=5)
    _registrar_fuente_ui(btn_email,
                         tamaño_offset=-1)
    btn_email.pack(side="left",
                   padx=3)

    _label_ui(tab_acerca,
              EMAIL_CONTACTO,
              tamaño_offset=-2,
              bg="#121212",
              fg="#777777").pack(pady=(5, 20))

    ttk.Separator(tab_acerca,
                  orient="horizontal").pack(fill="x",
                                            padx=40,
                                            pady=15)

    btn_info = tk.Button(tab_acerca,
                         text="ℹ️ Más información y créditos",
                         command=abrir_info_completa,
                         bg="#FF8800",
                         fg="#000000",
                         activebackground="#FFA033",
                         activeforeground="#000000",
                         relief="flat",
                         font=("Segoe UI", 11, "bold"),
                         padx=20,
                         pady=8)
    btn_info._color_fijo = True
    btn_info.pack(pady=10)

    
def abrir_info_completa():
    global _info_completa_ref

    if _info_completa_ref is not None and _info_completa_ref.winfo_exists():
        _info_completa_ref.lift()
        _info_completa_ref.focus_force()
        return

    info = tk.Toplevel(ventana)
    _info_completa_ref = info
    info.title("Acerca de Terpi")
    info.geometry("620x760")
    info.configure(bg="#121212")
    info.transient(ventana)
    info.grab_set()

    canvas = tk.Canvas(info,
                       bg="#121212",
                       highlightthickness=0,
                       bd=0)
    scroll = ttk.Scrollbar(info,
                           orient="vertical",
                           command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right",
                fill="y")
    canvas.pack(side="left",
                fill="both",
                expand=True)

    inner = tk.Frame(canvas,
                     bg="#121212")
    canvas.create_window((0, 0),
                         window=inner,
                         anchor="nw",
                         width=600)

    def _on_inner_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    inner.bind("<Configure>", _on_inner_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_destroy(event):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    info.bind("<Destroy>", _on_destroy)

    def _abrir_url(url):
        import webbrowser
        webbrowser.open(url)

    # 1. BIENVENIDA
    
    ruta_bienvenida = os.path.join(CARPETA_PROGRAMA, "assets", "ami_bienvenida.png")
    if os.path.exists(ruta_bienvenida):
        try:
            from PIL import Image, ImageTk
            img = Image.open(ruta_bienvenida)
            img.thumbnail((280, 280), Image.Resampling.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            lbl = tk.Label(inner, image=foto, bg="#121212", bd=0)
            lbl.image = foto
            lbl.pack(pady=(20, 10))
        except Exception as e:
            log.warning("No se pudo cargar ami_bienvenida.png: %s", e)

    _label_ui(inner, "Bienvenido/a a Terpi",
              tamaño_offset=+5,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(5, 15))

    # 2. FILOSOFÍA
    
    ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=40, pady=10)

    _label_ui(inner, "Filosofía", tamaño_offset=+2, estilo="bold",
              bg="#121212", fg="white").pack(pady=(10, 5))

    texto_filosofia = (
        "Terpi es freeware: gratis, para siempre. No hay versión "
        "\"premium\", no hay funciones bloqueadas, no hay trampa, "
        "ni compromiso. Terpi nace libre, para quien lo necesite. \n\n"
        "Es open source: el código está en GitHub y puedes estudiarlo, "
        "modificarlo o mejorarlo. Si haces una versión mejor, sólo pido que "
        "la compartas de manera gratuita, siguiendo el espíritu de Terpi. \n\n"
        "Es de intérpretes, para intérpretes: nació de la necesidad real "
        "de quienes trabajamos con audio en vivo; de mi según mis capacidades, "
        "a todos ustedes, de acuerdo a nuestras necesidades. \n\n"
    )
    _label_ui(inner,
              texto_filosofia,
              tamaño_offset=-2,
              bg="#121212",
              fg="#CCCCCC",
              justify="left",
              wraplength=500).pack(padx=40, pady=5)

    # 3. CRÉDITOS
    
    ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=40, pady=15)

    ruta_pulgar = os.path.join(CARPETA_PROGRAMA, "assets", "ami_pulgar.png")
    if os.path.exists(ruta_pulgar):
        try:
            from PIL import Image, ImageTk
            img = Image.open(ruta_pulgar)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            lbl = tk.Label(inner, image=foto, bg="#121212", bd=0)
            lbl.image = foto
            lbl.pack(pady=(10, 5))
        except Exception as e:
            log.warning("No se pudo cargar ami_pulgar.png: %s", e)

    _label_ui(inner,
              "¿Te gustó Terpi?",
              tamaño_offset=+2,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(5, 5))

    _label_ui(inner,
              "Terpi existe gracias a estas herramientas:",
              tamaño_offset=0,
              bg="#121212",
              fg="#CCCCCC").pack(pady=(0, 10))

    for nombre, creador, desc, lic in [
        ("Vosk", "Alpha Cephei", "Reconocimiento de voz offline", "Apache 2.0"),
        ("Argos Translate", "Argos Open Tech", "Traducción offline", "GPL-3.0"),
        ("Punctuators", "1-800-BAD-CODE", "Puntuación automática", "MIT"),
        ("pyaudiowpatch", "s0d3s", "Captura de audio WASAPI", "MIT"),
        ("Pillow", "Fredrik Lundh / Jeffrey A. Clark", "Procesamiento de imágenes", "HPND"),
        ("NumPy", "NumPy Developers", "Procesamiento numérico", "BSD-3-Clause"),
        ("Python", "Python Software Foundation", "El lenguaje", "PSF License"),
        ("Tkinter", "Python Software Foundation", "Interfaz gráfica", "PSF License"),
        ("Topaz A500", "rewtnull", "Fuente del tema TerpiAmiga", "GPL-FE"),
        ("Jost", "Owen Earl", "Fuente de la interfaz", "OFL"),
        ("VT323", "Peter Hull", "Fuente del tema Cassette Futurism", "OFL"),
        ("Merriweather", "Eben Sorkin", "Fuente del área de texto", "OFL"),
        ("VoxLingua107", "Autores originales: Jörgen Valk y Tanel Alumäe (SpeechBrain)", "Clasificador de voz", "CC BY 4.0"),
        ("VoxLingua107", "Conversion y optimización ONNX: Christopher Thompson (Vernacula)", "Clasificador de voz", "Apache 2.0"),
        ("Silero VAD", "Silero Team", "Modelo de detección de actividad vocal", "MIT"),
        ("ONNX Runtime", "Microsoft y coolaboradores", "Motor de inferencia local para modelos de IA", "MIT"),
        ("Terpi utiliza el modelo ONNX sin modificaciones", "", "", "")
    ]:
        fila = tk.Frame(inner,
                        bg="#121212")
        fila.pack(fill="x",
                  padx=40,
                  pady=1)
        _label_ui(fila,
                  f"• {nombre}",
                  tamaño_offset=-2, 
                  bg="#121212",
                  fg="#E0E0E0",
                  anchor="w").pack(side="left")
        _label_ui(fila, f" — {creador}",
                  tamaño_offset=-2, 
                  bg="#121212",
                  fg="#AAAAAA").pack(side="left")
        _label_ui(fila,
                  f" · {desc} ({lic})",
                  tamaño_offset=-2, 
                  bg="#121212",
                  fg="#777777").pack(side="left")

    # 4. AGRADECIMIENTOS
    
    ttk.Separator(inner,
                  orient="horizontal").pack(fill="x",
                                            padx=40,
                                            pady=15)

    _label_ui(inner,
              "Agradecimientos",
              tamaño_offset=+2,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(10, 5))

    texto_agradecimientos = (
        "A la comunidad de intérpretes en español, que recibió este "
        "proyecto con los brazos abiertos por lo que es, quienes me expresaron "
        "sus comentarios, y me ayudaron a formar Terpi un paso a la vez. Así"
        "como a los moderadores del grupo Memeline, que no me cerraron la puerta"
        "en la cara cuando publiqué Terpi la primera vez, cuando solo era Transcriptor \n\n"
        "A los desarrolladores de Vosk, Argos, Punctuators, PortAudio "
        "y todos los proyectos de código abierto que hacen posible que "
        "una sola persona, parada en hombros de gigantes, pueda construir algo así. \n\n"
        "A la Commodore Amiga y la cultura freeware de los 90s, que no solo "
        "me dieron programas que usé y disfruté de manera gratuita, si no que me "
        "infundieron el deseo de darle a la comunidad lo que la comunidad "
        "en su momento me dió a mi, así como darme una identidad visual. \n\n"
    )
    _label_ui(inner,
              texto_agradecimientos,
              tamaño_offset=-2,
              bg="#121212",
              fg="#CCCCCC",
              justify="left",
              wraplength=500).pack(padx=40, pady=5)

    # 5. DONACIONES
    def _formatear_clabe(clabe):
        clabe = clabe.replace(" ", "")
        if len(clabe) != 18:
            return clabe
        return f"{clabe[0:4]} {clabe[4:8]} {clabe[8:12]} {clabe[12:16]} {clabe[16:18]}"

    ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=40, pady=15)

    ruta_cajita = os.path.join(CARPETA_PROGRAMA, "assets", "ami_cajita.png")
    if os.path.exists(ruta_cajita):
        try:
            from PIL import Image, ImageTk
            img = Image.open(ruta_cajita)
            img.thumbnail((180, 180), Image.Resampling.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            lbl = tk.Label(inner, image=foto, bg="#121212", bd=0)
            lbl.image = foto
            lbl.pack(pady=(10, 5))
        except Exception as e:
            log.warning("No se pudo cargar ami_cajita.png: %s", e)

    _label_ui(inner,
              "¿Te resultó útil Terpi?",
              tamaño_offset=+2,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(5, 5))
    _label_ui(inner,
              "Es gratis y siempre lo será. Si quieres apoyar el desarrollo:",
              tamaño_offset=-1,
              bg="#121212",
              fg="#CCCCCC").pack(pady=(0, 10))

    frame_don = tk.Frame(inner,
                         bg="#121212")
    frame_don.pack(pady=5)

    _label_ui(inner, "Transferencia directa (SPEI):",
              tamaño_offset=-1,
              bg="#121212",
              fg="#CCCCCC").pack(pady=(15, 3))

    frame_clabe = tk.Frame(inner,
                           bg="#121212")
    frame_clabe.pack(pady=3)

    _label_ui(frame_clabe, (f"CLABE: {_formatear_clabe(CLABE_DONACIONES)}", ...),
              tamaño_offset=-1,
              bg="#121212",
              fg="#E0E0E0"
              ).pack(side="left", padx=5)

    def _copiar_clabe():
        try:
            info.clipboard_clear()
            info.clipboard_append(CLABE_DONACIONES)
            btn_copiar.config(text="✅ Copiado")
            info.after(2000, lambda: btn_copiar.config(text="Copiar"))
        except Exception as e:
            log.warning("Error copiando CLABE: %s", e)

    btn_copiar = tk.Button(frame_clabe,
                           text="Copiar",
                           command=_copiar_clabe,
                           bg="#2A2A2A",
                           fg="white",
                           activebackground="#3A3A3A",
                           activeforeground="white",
                           relief="flat",
                           padx=8,
                           pady=3)
    _registrar_fuente_ui(btn_copiar,
                         tamaño_offset=-2)
    btn_copiar.pack(side="left",
                    padx=5)

    _label_ui(inner, f"Beneficiario: Ivan Santiago Cacho Cordova",
              tamaño_offset=-2,
              bg="#121212",
              fg="#888888").pack(pady=(3, 15))
    frame_don = tk.Frame(inner, bg="#121212")
        
    frame_don.pack(pady=5)

    btn_paypal = tk.Button(frame_don,
                           text="Donar con PayPal",
                           command=lambda: _abrir_url(URL_PAYPAL),
                           bg="#0070BA",
                           fg="#FFFFFF",
                           activebackground="#005EA6",
                           activeforeground="#FFFFFF",
                           relief="flat",
                           font=("Segoe UI", 10, "bold"),
                           padx=20,
                           pady=8)
    btn_paypal._color_fijo = True
    btn_paypal.pack(side="left", padx=3)

    # 6. CIERRE
    
    ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=40, pady=15)

    ruta_cansada = os.path.join(CARPETA_PROGRAMA, "assets", "ami_cansada.png")
    if os.path.exists(ruta_cansada):
        try:
            from PIL import Image, ImageTk
            img = Image.open(ruta_cansada)
            img.thumbnail((320, 320), Image.Resampling.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            lbl = tk.Label(inner, image=foto, bg="#121212", bd=0)
            lbl.image = foto
            lbl.pack(pady=(10, 10))
        except Exception as e:
            log.warning("No se pudo cargar ami_cansada.png: %s", e)

    _label_ui(inner,
              "The horrors persist, but so do we.",
              tamaño_offset=+1,
              estilo="italic",
              bg="#121212",
              fg="#CCCCCC").pack(pady=(5, 15))

    # 7. DESPEDIDA

    _label_ui(inner,
              "¡Gracias por usar Terpi!",
              tamaño_offset=+3,
              estilo="bold",
              bg="#121212",
              fg="white").pack(pady=(10, 20))

    btn_cerrar = tk.Button(inner,
                           text="Cerrar",
                           command=info.destroy,
                           bg="#2A2A2A",
                           fg="white",
                           activebackground="#3A3A3A",
                           activeforeground="white",
                           relief="flat",
                           padx=20, pady=6)
    
    _registrar_fuente_ui(btn_cerrar, tamaño_offset=-1)
    btn_cerrar.pack(pady=(0, 30))



    _aplicar_colores_botones(info)

# ---------------------------------------------------------------------------
# Construcción de la UI
# ---------------------------------------------------------------------------
def ocultar_panel_superior():
    frame_panel_superior.pack_forget()
    label_estado.pack_forget()
    btn_mostrar_flotante.place(relx=1.0, rely=0.0, x=-10, y=10, anchor="ne")


def mostrar_panel_superior():
    btn_mostrar_flotante.place_forget()
    frame_panel_superior.pack(fill="x",
                              padx=0,
                              pady=0,
                              before=texto)
    label_estado.pack(anchor="w",
                      padx=20,
                      before=texto)


def alternar_panel_superior():
    if frame_panel_superior.winfo_ismapped():
        ocultar_panel_superior()
    else:
        mostrar_panel_superior()

def registrar_atajo_global():
    global _hotkey_thread

    if _hotkey_thread is not None:
        if _hotkey_thread.is_alive():
            return

    _hotkey_thread = threading.Thread(target=_bucle_atajo_global, name="AtajoGlobal", daemon=True)

    _hotkey_thread.start()

def _bucle_atajo_global():
    global _hotkey_registrado
    global _hotkey_thread_id

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    _hotkey_thread_id = (ctypes.windll.kernel32.GetCurrentThreadId())

    mensaje = wintypes.MSG()

    _hotkey_registrado = bool(user32.RegisterHotKey( None, HOTKEY_ID, MOD_NOREPEAT, VK_F8))

    if not _hotkey_registrado:
        log.warning( "No se pudo registrar F8. Error: %d", ctypes.get_last_error())
        _hotkey_thread_id = None
        return

    log.info("Atajo global F8 registrado")

    try:
        while True:
            resultado = user32.GetMessageW( ctypes.byref(mensaje), None, 0, 0)
            if resultado == 0:
                break
            if resultado == -1:
                log.error("Error en GetMessageW")
                break

            if (mensaje.message == WM_HOTKEY and mensaje.wParam == HOTKEY_ID):
                log.info("Pulsación global F8 detectada")
                cola_ui.put(("cambiar_idioma_manual",))

    finally:
        user32.UnregisterHotKey(None, HOTKEY_ID)
        _hotkey_registrado = False
        _hotkey_thread_id = None

        log.info("Atajo global liberado")

def liberar_atajo_global():
    if _hotkey_thread_id is not None:
        ctypes.windll.user32.PostThreadMessageW(_hotkey_thread_id, WM_QUIT, 0, 0)

    if _hotkey_thread is not None:
        _hotkey_thread.join(timeout=1.0)

def construir_interfaz_principal():
    global combo_fuente, btn_reiniciar, var_traduccion, chk_traduccion
    global var_puntuacion, chk_puntuacion, label_estado, texto, scroll_texto
    global frame_fuente_ref, frame_toggles_ref, frame_panel_superior
    global btn_mostrar_flotante
    if titulo_carga is not None:
        titulo_carga.destroy()
    if mensaje_carga is not None:
        mensaje_carga.destroy()
    if gif_carga is not None:
        gif_carga.destroy()

    
    ventana.bind("<Configure>",
                 lambda e: _on_resize_ventana() if e.widget == ventana else None)
    aplicar_fondo()

    frame_panel_superior = tk.Frame(ventana,
                                    bg=_config_visual["fondo_ventana"])
    frame_panel_superior.pack(fill="x",
                              padx=0,
                              pady=0)

    frame_fuente_ref = tk.Frame(frame_panel_superior,
                                bg=_config_visual["fondo_ventana"])
    frame_fuente_ref.pack(fill="x",
                          padx=20,
                          pady=(15, 5))

    lbl_fuente = tk.Label(frame_fuente_ref,
                          text="Fuente:",
                          bg=_config_visual["fondo_ventana"],
                          fg="#BBBBBB")
    _registrar_fuente_ui(lbl_fuente,
                         tamaño_offset=0)
    lbl_fuente.pack(side="left", padx=(0, 8))

    combo_fuente = ttk.Combobox(frame_fuente_ref,
                                state="readonly",
                                width=55,
                                font=(_config_visual["fuente_ui"], 10))
    combo_fuente.pack(side="left", padx=(0, 8))

    btn_reiniciar = tk.Button(frame_fuente_ref,
                              text="🔄 Refrescar (F5)",
                              bg="#2A2A2A",
                              fg="white",
                              activebackground="#3A3A3A",
                              activeforeground="white",
                              relief="flat")
    _registrar_fuente_ui(btn_reiniciar, tamaño_offset=-1)
    btn_reiniciar.pack(side="left")

    btn_ajustes = tk.Button(frame_fuente_ref,
                            text="⚙️",
                            bg="#2A2A2A",
                            fg="white",
                            activebackground="#3A3A3A",
                            activeforeground="white",
                            relief="flat",
                            command=abrir_ventana_ajustes)
    _registrar_fuente_ui(btn_ajustes, tamaño_offset=+1)
    btn_ajustes.pack(side="left", padx=(10, 0))

    btn_ocultar = tk.Button(frame_fuente_ref,
                            text="👁 Ocultar",
                            bg="#2A2A2A",
                            fg="white",
                            activebackground="#3A3A3A",
                            activeforeground="white",
                            relief="flat",
                            command=ocultar_panel_superior)
    _registrar_fuente_ui(btn_ocultar, tamaño_offset=-1)
    btn_ocultar.pack(side="right")

    frame_toggles_ref = tk.Frame(frame_panel_superior,
                                 bg=_config_visual["fondo_ventana"])
    frame_toggles_ref.pack(fill="x",
                           padx=20,
                           pady=(5, 5))

    var_traduccion = tk.BooleanVar(value=True)

    def alternar_traduccion():
        global TRADUCCION_ACTIVA
        TRADUCCION_ACTIVA = var_traduccion.get()
        chk_traduccion.config(text="🌐 Traducción (F6): ON" if TRADUCCION_ACTIVA else "🌐 Traducción (F6): OFF")

    chk_traduccion = tk.Checkbutton(frame_toggles_ref,
                                    text="🌐 Traducción (F6): ON",
                                    variable=var_traduccion,
                                    command=alternar_traduccion,
                                    bg=_config_visual["fondo_ventana"],
                                    fg="#BBBBBB",
                                    activebackground=_config_visual["fondo_ventana"],
                                    activeforeground="white",
                                    selectcolor="#2A2A2A",
                                    relief="flat",
                                    highlightthickness=0)
    _registrar_fuente_ui(chk_traduccion, tamaño_offset=-1)
    chk_traduccion.pack(side="left", padx=(0, 12))

    var_puntuacion = tk.BooleanVar(value=True)

    def alternar_puntuacion():
        global PUNTUACION_ACTIVA
        PUNTUACION_ACTIVA = var_puntuacion.get()
        chk_puntuacion.config(text="✒️ Puntuación (F7): ON" if PUNTUACION_ACTIVA else "✒️ Puntuación (F7): OFF")

    chk_puntuacion = tk.Checkbutton(frame_toggles_ref,
                                    text="✒️ Puntuación (F7): ON",
                                    variable=var_puntuacion,
                                    command=alternar_puntuacion,
                                    bg=_config_visual["fondo_ventana"],
                                    fg="#BBBBBB",
                                    activebackground=_config_visual["fondo_ventana"],
                                    activeforeground="white",
                                    selectcolor="#2A2A2A",
                                    relief="flat",
                                    highlightthickness=0)
    _registrar_fuente_ui(chk_puntuacion, tamaño_offset=-1)
    chk_puntuacion.pack(side="left")

    var_auto_id = tk.BooleanVar(value=_config_visual.get("deteccion_automatica_idioma", True))

    def alternar_auto_id():
        activo = var_auto_id.get()
        establecer_deteccion_automatica(activo)
        chk_auto_id.config(text="Auto ID: ON" if activo else "Auto ID: OFF")
    chk_auto_id = tk.Checkbutton(frame_toggles_ref,
                                 text="Auto ID: ON" if var_auto_id.get() else "Auto ID: OFF",
                                 variable=var_auto_id,
                                 command=alternar_auto_id,
                                 bg=_config_visual["fondo_ventana"],
                                 fg="#BBBBBB",
                                 activebackground=_config_visual["fondo_ventana"],
                                 activeforeground="white",
                                 selectcolor="#2A2A2A",
                                 relief="flat",
                                 highlightthickness=0)

    _registrar_fuente_ui(chk_auto_id, tamaño_offset=-1)
    chk_auto_id.pack(side="left", padx=(12,0))

    label_estado = tk.Label(ventana,
                            text="",
                            bg=_config_visual["fondo_ventana"],
                            fg="#888888")
    _registrar_fuente_ui(label_estado, tamaño_offset=-1)
    label_estado.pack(anchor="w", padx=20)

    texto = tk.Text(ventana,
                    font=(_config_visual["fuente_texto"],
                          _config_visual["tamaño_texto"]),
                    wrap="word",
                    state="disabled",
                    bg=_config_visual["fondo_texto"],
                    fg="white",
                    padx=15,
                    pady=10)
    
    scroll_texto = tk.Scrollbar(ventana, command=texto.yview)
    texto.config(yscrollcommand=scroll_texto.set)
    scroll_texto.pack(side="right",
                      fill="y",
                      padx=(0, 20),
                      pady=(5, 15))
    texto.pack(fill="both",
               expand=True,
               padx=(20, 0),
               pady=(5, 15))

    texto.mark_set("parcial_inicio", "end-1c")
    texto.mark_gravity("parcial_inicio", tk.LEFT)

    aplicar_tema_visual()

    btn_mostrar_flotante = tk.Button(ventana,
                                     text="👁",
                                     bg="#2A2A2A",
                                     fg="white",
                                     relief="flat",
                                     command=mostrar_panel_superior)
    _registrar_fuente_ui(btn_mostrar_flotante, tamaño_offset=-2)

    combo_fuente.bind("<<ComboboxSelected>>", al_cambiar_fuente)
    btn_reiniciar.config(command=refrescar_lista_dispositivos)

    preparar_reconocedores(SR_ACTUAL)
    ventana.bind("<space>", cambiar_idioma)
    ventana.bind("<F2>", limpiar_conversacion)
    ventana.bind("<F5>", lambda e: refrescar_lista_dispositivos())
    ventana.bind("<F6>", alternar_traduccion_teclado)
    ventana.bind("<F7>", alternar_puntuacion_teclado)
    registrar_atajo_global()

    ventana.after(0, refrescar_lista_dispositivos)
    ventana.after(TICK_PROCESAR_MS, procesar_audio)
    ventana.after(TICK_WATCHDOG_MS, vigilar_stream)
    ventana.after(TICK_BUFFER_MS, revisar_buffer)
    ventana.after(50, _procesar_cola_ui)


# ---------------------------------------------------------------------------
# Cierre de app
# ---------------------------------------------------------------------------
def _cerrar_aplicacion():
    log.info("Cerrando Terpi...")
    try:
        try:
            _config_visual["geometria_ventana"] = ventana.geometry()
        except Exception as e:
            log.warning("No se pudo leer la geometría: %s", e)

        try:
            for job in ventana.tk.call("after", "info"):
                ventana.after_cancel(job)
        except Exception:
            pass

        cerrar_stream()

        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        _executor.shutdown(wait=False, cancel_futures=True)
        _executor_voxlingua.shutdown(wait=False, cancel_futures=False)

        liberar_atajo_global()
        guardar_configuracion()
    except Exception as e:
        log.exception("Error durante el cierre: %s", e)
    finally:
        try:
            ventana.destroy()
        except Exception:
            pass
        os._exit(0)


ventana.protocol("WM_DELETE_WINDOW", _cerrar_aplicacion)


# ---------------------------------------------------------------------------
# Inicio
# ---------------------------------------------------------------------------

hilo_carga = threading.Thread(target=proceso_de_carga, daemon=True)
hilo_carga.start()

ventana.mainloop()
