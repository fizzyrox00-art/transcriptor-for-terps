import os
import sys
import logging

ID_ARGOS_ES_EN = "1xrgoRlXscIBXNvhHWKIKANPYdWmLl4br"

NOMBRE_ARGOS_ES_EN = "translate-es_en.argosmodel"

if os.name == "nt":
    CARPETA_ARGOS = os.path.join(
        os.environ.get("USERPROFILE", os.path.expanduser("~")),
        ".local", "share", "argos-translate", "packages"
    )
else:
    CARPETA_ARGOS = os.path.join(
        os.path.expanduser("~"),
        ".local", "share", "argos-translate", "packages"
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def _importar_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        log.info("gdown no encontrado. Instalando...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "gdown"
        ])
        import gdown
        return gdown


def descargar_modelo_argos():
    """
    Descarga e instala el modelo de Argos es→en desde Google Drive.
    Devuelve True si tuvo éxito.
    """
    os.makedirs(CARPETA_ARGOS, exist_ok=True)

    try:
        import argostranslate.translate
        idiomas = argostranslate.translate.get_installed_languages()
        codigos = {lang.code for lang in idiomas}
        if "es" in codigos and "en" in codigos:
            es = next(l for l in idiomas if l.code == "es")
            en = next(l for l in idiomas if l.code == "en")
            if es.get_translation(en) is not None:
                log.info("Modelo Argos es→en ya instalado.")
                return True
    except Exception:
        pass

    gdown = _importar_gdown()
    ruta_modelo = os.path.join(CARPETA_ARGOS, NOMBRE_ARGOS_ES_EN)

    if os.path.exists(ruta_modelo):
        tamano = os.path.getsize(ruta_modelo)
        if tamano > 1_000_000:
            log.info("Modelo ya descargado (%.1f MB). Instalando...",
                     tamano / 1_000_000)
        else:
            os.remove(ruta_modelo)

    if not os.path.exists(ruta_modelo):
        log.info("Descargando modelo Argos es→en (~100 MB)...")
        url = f"https://drive.google.com/uc?id={ID_ARGOS_ES_EN}"
        try:
            resultado = gdown.download(url, ruta_modelo, quiet=False)
        except Exception as e:
            log.error("Error descargando el modelo: %s", e)
            return False

        if resultado is None or not os.path.exists(ruta_modelo):
            log.error("No se pudo descargar el modelo.")
            return False

        tamano = os.path.getsize(ruta_modelo)
        if tamano < 1_000_000:
            log.error("Archivo descargado demasiado pequeño (%.1f MB). "
                      "Probablemente sea una página de error de Drive.",
                      tamano / 1_000_000)
            os.remove(ruta_modelo)
            return False

        log.info("Modelo descargado (%.1f MB).", tamano / 1_000_000)

    try:
        import argostranslate.package
        argostranslate.package.install_from_path(ruta_modelo)
        log.info("Modelo Argos es→en instalado correctamente.")
        return True
    except Exception as e:
        log.error("No se pudo instalar el modelo en Argos: %s", e)
        return False


if __name__ == "__main__":
    exito = descargar_modelo_argos()
    sys.exit(0 if exito else 1)
