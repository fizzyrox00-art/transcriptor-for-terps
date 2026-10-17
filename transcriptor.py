import tkinter as tk
from tkinter import ttk
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
import numpy as np
import pyaudiowpatch as pyaudio
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
os.environ.setdefault("ARGOS_INTRA_THREADS", "4")
os.environ.setdefault("ARGOS_INTER_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
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
# Manejo de excepciones fatales
# ---------------------------------------------------------------------------
def _manejar_excepcion_fatal(tipo, valor, tb):
    """Captura cualquier excepción no manejada y pausa antes de cerrar."""
    traceback.print_exception(tipo, valor, tb)
    print()
    print("=" * 60)
    print("El programa encontró un error y se cerrará.")
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
TIMEOUT_FLUSH_SEG = 2.0

BLOQUES_POR_CICLO = 5
TICK_PROCESAR_MS = 10
TICK_WATCHDOG_MS = 2000
TICK_BUFFER_MS = 500
TIMEOUT_STREAM_SEG = 5

CODIGO_ARGOS = {"INGLÉS": "en", "ESPAÑOL": "es"}

CACHE_TRADUCCIONES = 3000
MIN_PALABRAS_PUNTUAR = 4
HILOS_PROCESAMIENTO = 1


# ---------------------------------------------------------------------------
# Ventana
# ---------------------------------------------------------------------------
ventana = tk.Tk()
ventana.geometry("900x650")
ventana.configure(bg="#000000")
ventana.title("[EN] Transcriptor")


titulo_carga = tk.Label(
    ventana, text="Transcriptor", font=("Georgia", 24),
    bg="#000000", fg="white"
)
titulo_carga.pack(pady=(180, 20))

mensaje_carga = tk.Label(
    ventana, text="Iniciando...", font=("Georgia", 12),
    bg="#000000", fg="#BBBBBB"
)
mensaje_carga.pack()
ventana.update()


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

_ultimo_parcial = ""

audio_queue = queue.Queue(maxsize=MAX_COLA_AUDIO)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=HILOS_PROCESAMIENTO)

stream_actual = {
    "stream": None,
    "pyaudio": None,
    "device": None,
    "sr": None,
    "canales": 1,
    "ultimo_bloque": time.time(),
}

_dispositivos_cache = []
_primera_carga = True

_estadisticas_cache = {"hits": 0, "misses": 0}


# ---------------------------------------------------------------------------
# Puntuación
# ---------------------------------------------------------------------------
def cargar_puntuador():
    global _puntuador
    try:
        mensaje_carga.config(text="Cargando modelo de puntuación...")
        ventana.update()
        t0 = time.perf_counter()
        _puntuador = PunctCapSegModelONNX.from_pretrained("pcs_47lang")
        t1 = time.perf_counter()
        log.info("Puntuador cargado en %.2f s", t1 - t0)
    except Exception as e:
        log.error("No se pudo cargar el puntuador: %s", e)
        _puntuador = None


def _normalizar_palabra(p):
    return p.strip(".,?!;:\"-—–()[]{}«»“”‘’¿¡").lower()


def puntuar(texto):
    if not PUNTUACION_ACTIVA or _puntuador is None:
        return texto
    if not texto.strip():
        return texto
    if len(texto.split()) < MIN_PALABRAS_PUNTUAR:
        return texto

    try:
        resultados = _puntuador.infer([texto])
        if not resultados or not resultados[0]:
            return texto
        texto_final = " ".join(resultados[0]).strip()
    except Exception as e:
        log.warning("Error en puntuación: %s", e)
        return texto

    if not texto_final:
        return texto

    palabras_original = set(_normalizar_palabra(w) for w in texto.split())
    palabras_final = set(_normalizar_palabra(w) for w in texto_final.split())
    perdidas = palabras_original - palabras_final
    if perdidas:
        log.debug("Puntuador perdió palabras: %s. Usando original.", perdidas)
        return texto

    return texto_final


# ---------------------------------------------------------------------------
# Verificación de modelos de traducción (es→en desde Drive)
# ---------------------------------------------------------------------------
def verificar_y_descargar_modelo_es_en():
    try:
        import argostranslate.translate
        idiomas = argostranslate.translate.get_installed_languages()
        codigos = {lang.code for lang in idiomas}

        if "es" in codigos and "en" in codigos:
            es = next(l for l in idiomas if l.code == "es")
            en = next(l for l in idiomas if l.code == "en")
            try:
                traduccion = es.get_translation(en)
                if traduccion is not None:
                    log.info("Modelo es→en ya instalado.")
                    return True
            except Exception:
                pass

        log.info("Modelo es→en no encontrado. Iniciando descarga...")
        mensaje_carga.config(text="Descargando modelo es→en...")
        ventana.update()

        import descargar_modelos
        exito = descargar_modelos.descargar_modelo_argos()

        if exito:
            log.info("Modelo es→en descargado e instalado.")
            return True
        else:
            log.warning("No se pudo descargar el modelo es→en.")
            return False

    except Exception as e:
        log.error("Error verificando/descargando modelo es→en: %s", e)
        return False


# ---------------------------------------------------------------------------
# Traducción
# ---------------------------------------------------------------------------
def inicializar_traduccion():
    # Solo descargamos en→es desde PyPI.
    # El es→en lo gestiona descargar_modelos.py desde Google Drive.
    pares_a_descargar = [("en", "es")]

    # Pero SÍ queremos preparar los dos pares como traductores disponibles.
    pares_a_preparar = [("en", "es"), ("es", "en")]

    if os.path.isdir(CARPETA_TRADUCCIONES):
        archivos_locales = [
            f for f in os.listdir(CARPETA_TRADUCCIONES)
            if f.endswith(".argosmodel")
        ]
        for archivo in archivos_locales:
            ruta = os.path.join(CARPETA_TRADUCCIONES, archivo)
            log.info("Instalando modelo local: %s", archivo)
            mensaje_carga.config(text=f"Instalando {archivo}...")
            ventana.update()
            try:
                argostranslate.package.install_from_path(ruta)
            except Exception as e:
                log.warning("No se pudo instalar %s: %s", archivo, e)

    paquetes_instalados = argostranslate.package.get_installed_packages()
    for from_code, to_code in pares_a_descargar:
        ya = any(p.from_code == from_code and p.to_code == to_code
                 for p in paquetes_instalados)
        if ya:
            continue
        log.info("Falta %s→%s. Intentando descargar...", from_code, to_code)
        mensaje_carga.config(
            text=f"Descargando modelo {from_code}→{to_code}..."
        )
        ventana.update()
        try:
            argostranslate.package.update_package_index()
            disponibles = argostranslate.package.get_available_packages()
            paquete = next(
                (p for p in disponibles
                 if p.from_code == from_code and p.to_code == to_code),
                None
            )
            if paquete is None:
                continue
            ruta = paquete.download()
            argostranslate.package.install_from_path(ruta)
        except Exception as e:
            log.error("Error instalando %s→%s: %s", from_code, to_code, e)

    installed_languages = argostranslate.translate.get_installed_languages()
    idiomas = {lang.code: lang for lang in installed_languages}

    # Aquí iteramos sobre pares_a_preparar, no pares_a_descargar.
    for from_code, to_code in pares_a_preparar:
        if from_code in idiomas and to_code in idiomas:
            try:
                _traducciones[(from_code, to_code)] = (
                    idiomas[from_code].get_translation(idiomas[to_code])
                )
                log.info("Traducción %s→%s lista.", from_code, to_code)
            except Exception as e:
                log.error("No se pudo preparar %s→%s: %s",
                          from_code, to_code, e)
        else:
            log.warning("Falta el par %s→%s para preparar traductor.",
                        from_code, to_code)

    for (from_code, to_code), traductor in _traducciones.items():
        try:
            mensaje_carga.config(
                text=f"Precalentando {from_code}→{to_code}..."
            )
            ventana.update()
            muestra = "hello" if from_code == "en" else "hola"
            _ = traductor.translate(muestra)
        except Exception as e:
            log.warning("No se pudo precalentar %s→%s: %s",
                        from_code, to_code, e)


@lru_cache(maxsize=CACHE_TRADUCCIONES)
def _traducir_cacheado(texto, from_code, to_code):
    traductor = _traducciones.get((from_code, to_code))
    if traductor is None:
        return None
    try:
        return traductor.translate(texto)
    except Exception as e:
        log.warning("Error traduciendo: %s", e)
        return None


def traducir(texto):
    if not TRADUCCION_ACTIVA:
        return None
    from_code = CODIGO_ARGOS.get(IDIOMA)
    to_code = "es" if from_code == "en" else "en"
    info_antes = _traducir_cacheado.cache_info()
    resultado = _traducir_cacheado(texto, from_code, to_code)
    info_despues = _traducir_cacheado.cache_info()
    if info_despues.hits > info_antes.hits:
        _estadisticas_cache["hits"] += 1
    else:
        _estadisticas_cache["misses"] += 1
    return resultado


# ---------------------------------------------------------------------------
# Detección de dispositivos de loopback (WASAPI)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Reconocedores Vosk
# ---------------------------------------------------------------------------
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
                log.info("Endpointer modo LONG activado para %s.", idioma)
            except Exception as e:
                log.warning("No se pudo activar EndpointerMode.LONG: %s", e)

        _reconocedores[idioma] = rec
    log.info("Reconocedores preparados @ %d Hz", sr)


def cargar_modelos():
    for idioma, ruta in RUTAS_MODELOS.items():
        if _modelos[idioma] is None:
            mensaje_carga.config(text=f"Cargando modelo {idioma.lower()}...")
            ventana.update()
            log.info("Cargando modelo %s desde %s", idioma, ruta)
            _modelos[idioma] = Model(ruta)
            log.info("Modelo %s cargado.", idioma)


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


# ---------------------------------------------------------------------------
# Callback de audio (PyAudioWPatch)
# ---------------------------------------------------------------------------
def callback_pyaudio(in_data, frame_count, time_info, status):
    if status:
        log.warning("Audio status: %s", status)

    stream_actual["ultimo_bloque"] = time.time()

    try:
        audio_np = np.frombuffer(in_data, dtype=np.int16)

        canales = stream_actual.get("canales", 1)
        if canales > 1:
            audio_np = (audio_np.reshape(-1, canales)
                        .mean(axis=1)
                        .astype(np.int16))

        audio_queue.put_nowait(audio_np.tobytes())
    except queue.Full:
        pass
    except Exception as e:
        log.warning("Error en callback: %s", e)

    return (None, pyaudio.paContinue)


# ---------------------------------------------------------------------------
# Control del stream
# ---------------------------------------------------------------------------
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
            format=pyaudio.paInt16,
            channels=canales,
            rate=sr,
            frames_per_buffer=blocksize,
            input=True,
            input_device_index=indice,
            stream_callback=callback_pyaudio,
        )
        stream.start_stream()

        stream_actual["stream"] = stream
        stream_actual["pyaudio"] = p
        stream_actual["device"] = indice
        stream_actual["sr"] = sr
        stream_actual["ultimo_bloque"] = time.time()
        log.info("Stream loopback abierto: [%d] %s @ %d Hz, %d canales",
                 indice, dispositivo_dict["nombre"], sr, canales)
    except Exception as e:
        log.error("Error abriendo stream: %s", e)
        raise


# ---------------------------------------------------------------------------
# Cargar todo
# ---------------------------------------------------------------------------
cargar_modelos()

mensaje_carga.config(text="Preparando traducción offline...")
ventana.update()

verificar_y_descargar_modelo_es_en()

inicializar_traduccion()

cargar_puntuador()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
titulo_carga.destroy()
mensaje_carga.destroy()

# --- Fila 1: selector de dispositivo ---
frame_fuente = tk.Frame(ventana, bg="#000000")
frame_fuente.pack(fill="x", padx=20, pady=(15, 5))

tk.Label(
    frame_fuente, text="Fuente:", font=("Georgia", 11),
    bg="#000000", fg="#BBBBBB"
).pack(side="left", padx=(0, 8))

combo_fuente = ttk.Combobox(
    frame_fuente, state="readonly", width=55, font=("Georgia", 10)
)
combo_fuente.pack(side="left", padx=(0, 8))

btn_reiniciar = tk.Button(
    frame_fuente, text="🔄 Refrescar lista", font=("Georgia", 10),
    bg="#2A2A2A", fg="white",
    activebackground="#3A3A3A", activeforeground="white", relief="flat"
)
btn_reiniciar.pack(side="left")


# --- Fila 2: toggles ---
frame_toggles = tk.Frame(ventana, bg="#000000")
frame_toggles.pack(fill="x", padx=20, pady=(5, 5))

var_traduccion = tk.BooleanVar(value=True)


def alternar_traduccion():
    global TRADUCCION_ACTIVA
    TRADUCCION_ACTIVA = var_traduccion.get()
    chk_traduccion.config(
        text="🌐 Traducción: ON" if TRADUCCION_ACTIVA
        else "🌐 Traducción: OFF"
    )
    log.info("Traducción %s",
             "ACTIVADA" if TRADUCCION_ACTIVA else "DESACTIVADA")


chk_traduccion = tk.Checkbutton(
    frame_toggles, text="🌐 Traducción: ON", variable=var_traduccion,
    command=alternar_traduccion, font=("Georgia", 10),
    bg="#000000", fg="#BBBBBB", activebackground="#000000",
    activeforeground="white", selectcolor="#2A2A2A",
    relief="flat", highlightthickness=0,
)
chk_traduccion.pack(side="left", padx=(0, 12))

var_puntuacion = tk.BooleanVar(value=True)


def alternar_puntuacion():
    global PUNTUACION_ACTIVA
    PUNTUACION_ACTIVA = var_puntuacion.get()
    chk_puntuacion.config(
        text="✒️ Puntuación: ON" if PUNTUACION_ACTIVA
        else "✒️ Puntuación: OFF"
    )
    log.info("Puntuación %s",
             "ACTIVADA" if PUNTUACION_ACTIVA else "DESACTIVADA")


chk_puntuacion = tk.Checkbutton(
    frame_toggles, text="✒️ Puntuación: ON", variable=var_puntuacion,
    command=alternar_puntuacion, font=("Georgia", 10),
    bg="#000000", fg="#BBBBBB", activebackground="#000000",
    activeforeground="white", selectcolor="#2A2A2A",
    relief="flat", highlightthickness=0,
)
chk_puntuacion.pack(side="left")

label_estado = tk.Label(
    ventana, text="", font=("Georgia", 10), bg="#000000", fg="#888888"
)
label_estado.pack(anchor="w", padx=20)


# ---------------------------------------------------------------------------
# Área de texto
# ---------------------------------------------------------------------------
texto = tk.Text(
    ventana, font=("Georgia", 11), wrap="word",
    state="disabled", bg="#1E1E1E", fg="white", padx=15, pady=10,
)

texto.tag_configure("ingles", foreground="#a5d7f2",
                    lmargin1=0, lmargin2=50)
texto.tag_configure("español", foreground="#f2b27a",
                    lmargin1=0, lmargin2=50)
texto.tag_configure("traduccion", foreground="#999999",
                    lmargin1=50, lmargin2=100)
texto.tag_configure("parcial", foreground="#B8B8B8",
                    font=("Georgia", 11, "italic"),
                    lmargin1=0, lmargin2=50)

scroll_texto = tk.Scrollbar(ventana, command=texto.yview)
texto.config(yscrollcommand=scroll_texto.set)
scroll_texto.pack(side="right", fill="y", padx=(0, 20), pady=(5, 15))
texto.pack(fill="both", expand=True, padx=(20, 0), pady=(5, 15))

texto.mark_set("parcial_inicio", "end-1c")
texto.mark_gravity("parcial_inicio", tk.LEFT)


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------
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
            texto.insert("end", entrada["traduccion"] + "\n", "traduccion")
        texto.insert("end", "\n")
    texto.insert("end", "\n")
    texto.mark_set("parcial_inicio", "end-1c")
    texto.mark_gravity("parcial_inicio", tk.LEFT)
    texto.config(state="disabled")
    texto.see("end")


def mostrar_parcial(contenido):
    texto.config(state="normal")
    texto.delete("parcial_inicio", "end")
    if contenido:
        _, prefijo = _etiqueta_y_prefijo(IDIOMA)
        texto.insert("end", f"{prefijo}{contenido}…\n", "parcial")
    texto.config(state="disabled")
    texto.see("end")


def mostrar_final(contenido):
    _, prefijo = _etiqueta_y_prefijo(IDIOMA)
    etiqueta_orig = "ingles" if IDIOMA == "INGLÉS" else "español"

    texto.config(state="normal")
    texto.delete("parcial_inicio", "end")
    texto.config(state="disabled")

    if DEBUG_TEXTO:
        log.info("[DEBUG] mostrar_final: '%s'", contenido)

    entrada = {
        "original": prefijo + contenido,
        "etiqueta": etiqueta_orig,
        "traduccion": None,
        "idioma_origen": IDIOMA,
        "contenido_crudo": contenido,
    }
    historial.append(entrada)
    _renderizar_todo()
    _executor.submit(_procesar_entrada, entrada)


def _procesar_entrada(entrada):
    try:
        contenido = entrada["contenido_crudo"]

        if DEBUG_TEXTO:
            log.info("[DEBUG] ANTES puntuar: '%s'", contenido)

        t0 = time.perf_counter()
        texto_puntuado = puntuar(contenido)
        t1 = time.perf_counter()

        if DEBUG_TEXTO:
            log.info("[DEBUG] DESPUÉS puntuar: '%s'", texto_puntuado)

        trad = traducir(texto_puntuado)
        t2 = time.perf_counter()
        log.debug("Puntuación=%.0f ms, traducción=%.0f ms",
                  (t1 - t0) * 1000, (t2 - t1) * 1000)
    except Exception as e:
        log.warning("Error procesando entrada: %s", e)
        return
    ventana.after(0, lambda: _aplicar_resultado(entrada, texto_puntuado, trad))


def _aplicar_resultado(entrada, texto_puntuado, trad):
    _, prefijo = _etiqueta_y_prefijo(entrada["idioma_origen"])
    entrada["original"] = prefijo + texto_puntuado
    if trad:
        sufijo = _sufijo_traduccion(entrada["idioma_origen"])
        entrada["traduccion"] = f"{' ' * 5}{sufijo}: {trad}"
    _renderizar_todo()


# ---------------------------------------------------------------------------
# Buffer de frases
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Filtros anti-alucinación
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Selector de dispositivos
# ---------------------------------------------------------------------------
def _aplicar_dispositivos(dispositivos):
    global _dispositivos_cache, _primera_carga
    _dispositivos_cache = dispositivos

    etiquetas = []
    for d in dispositivos:
        marca = " ⭐" if d["es_defecto"] else ""
        etiquetas.append(
            f"🔊 [{d['indice']:3d}] {d['nombre']}{marca}  "
            f"({d['sr']} Hz, {d['canales']} canales)"
        )
    combo_fuente["values"] = etiquetas

    if not dispositivos:
        label_estado.config(
            text="❌ No se encontraron dispositivos de loopback (WASAPI)."
        )
        return

    if _primera_carga:
        _primera_carga = False
        mejor = dispositivos[0]
        combo_fuente.current(0)
        label_estado.config(
            text=f"✅ {mejor['nombre']} @ {mejor['sr']} Hz"
        )
        try:
            abrir_stream(mejor)
        except Exception as e:
            log.error("Error abriendo stream inicial: %s", e)
            label_estado.config(text=f"❌ Error: {e}")
    else:
        label_estado.config(
            text=f"🔄 Lista actualizada ({len(dispositivos)} dispositivos)."
        )


def refrescar_lista_dispositivos():
    label_estado.config(
        text="⏳ Buscando dispositivos de loopback... (Espere un segundo)"
    )
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
        label_estado.config(
            text=f"✅ {d['nombre']} @ {d['sr']} Hz"
        )
    except Exception as e:
        log.error("Error cambiando de fuente: %s", e)
        label_estado.config(text=f"❌ Error: {e}")


combo_fuente.bind("<<ComboboxSelected>>", al_cambiar_fuente)
btn_reiniciar.config(command=refrescar_lista_dispositivos)


# ---------------------------------------------------------------------------
# Atajos
# ---------------------------------------------------------------------------
def cambiar_idioma(event):
    global IDIOMA, _ultimo_parcial

    texto_pendiente = flush_reconocedor_actual()
    _flush_buffer(forzar=True)

    rec_actual = _reconocedores.get(IDIOMA)
    if rec_actual is not None:
        rec_actual.Reset()

    if texto_pendiente:
        _confirmar_con_buffer(texto_pendiente)

    _ultimo_parcial = ""

    if IDIOMA == "INGLÉS":
        IDIOMA = "ESPAÑOL"
        ventana.title("[ES] Transcriptor")
    else:
        IDIOMA = "INGLÉS"
        ventana.title("[EN] Transcriptor")

    texto.config(state="normal")
    texto.delete("parcial_inicio", "end")
    texto.config(state="disabled")
    log.info("Idioma cambiado a: %s", IDIOMA)


def limpiar_conversacion(event):
    global historial, _buffer_frase, _ultimo_parcial
    historial = []
    _buffer_frase = ""
    _ultimo_parcial = ""
    for rec in _reconocedores.values():
        if rec is not None:
            rec.Reset()
    _renderizar_todo()

    info = _traducir_cacheado.cache_info()
    log.info("Caché: %d hits / %d misses (%.1f%%) | Tamaño: %d/%d",
             info.hits, info.misses,
             100 * info.hits / max(info.hits + info.misses, 1),
             info.currsize, info.maxsize)
    log.info("Conversación limpiada.")


def alternar_traduccion_teclado(event=None):
    var_traduccion.set(not var_traduccion.get())
    alternar_traduccion()


def alternar_puntuacion_teclado(event=None):
    var_puntuacion.set(not var_puntuacion.get())
    alternar_puntuacion()


ventana.bind("<space>", cambiar_idioma)
ventana.bind("<F2>", limpiar_conversacion)
ventana.bind("<F5>", lambda e: refrescar_lista_dispositivos())
ventana.bind("<F6>", alternar_traduccion_teclado)
ventana.bind("<F7>", alternar_puntuacion_teclado)


# ---------------------------------------------------------------------------
# Bucle de procesamiento
# ---------------------------------------------------------------------------
def procesar_audio():
    global _ultimo_parcial

    try:
        rec = _reconocedores.get(IDIOMA)
        if rec is not None:
            for _ in range(BLOQUES_POR_CICLO):
                try:
                    data = audio_queue.get_nowait()
                except queue.Empty:
                    break

                if rec.AcceptWaveform(data):
                    rec.AcceptWaveform(b"")
                    try:
                        resultado = json.loads(rec.FinalResult())
                    except Exception as e:
                        log.warning("JSON inválido: %s", e)
                        continue

                    texto_final = resultado.get("text", "").strip()

                    if _ultimo_parcial:
                        palabras_final = len(texto_final.split())
                        palabras_parcial = len(_ultimo_parcial.split())
                        if palabras_parcial > palabras_final:
                            if DEBUG_TEXTO:
                                log.info(
                                    "[DEBUG] Parcial (%d) > Final (%d). "
                                    "Usando parcial: '%s'",
                                    palabras_parcial, palabras_final,
                                    _ultimo_parcial[:80]
                                )
                            texto_final = _ultimo_parcial

                    _ultimo_parcial = ""

                    if texto_final and not es_alucinacion(texto_final, resultado):
                        _confirmar_con_buffer(texto_final)
                else:
                    try:
                        parcial = json.loads(rec.PartialResult())
                    except Exception:
                        continue
                    texto_parcial = parcial.get("partial", "")
                    if texto_parcial:
                        _ultimo_parcial = texto_parcial
                        mostrar_parcial(texto_parcial)
    except Exception as e:
        log.exception("Error en procesar_audio: %s", e)
    finally:
        ventana.after(TICK_PROCESAR_MS, procesar_audio)


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------
def vigilar_stream():
    try:
        if stream_actual["stream"] is not None:
            delta = time.time() - stream_actual["ultimo_bloque"]
            if delta > TIMEOUT_STREAM_SEG:
                log.warning("Stream sin audio hace %.1fs. Reiniciando...", delta)
                try:
                    dev_actual = next(
                        (d for d in _dispositivos_cache
                         if d["indice"] == stream_actual["device"]),
                        None
                    )
                    if dev_actual:
                        abrir_stream(dev_actual)
                except Exception as e:
                    log.error("No se pudo reiniciar stream: %s", e)
    except Exception as e:
        log.exception("Error en vigilar_stream: %s", e)
    finally:
        ventana.after(TICK_WATCHDOG_MS, vigilar_stream)


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
def arranque():
    preparar_reconocedores(SR_ACTUAL)
    ventana.after(0, refrescar_lista_dispositivos)
    ventana.after(TICK_PROCESAR_MS, procesar_audio)
    ventana.after(TICK_WATCHDOG_MS, vigilar_stream)
    ventana.after(TICK_BUFFER_MS, revisar_buffer)


ventana.after(100, arranque)


ventana.mainloop()
