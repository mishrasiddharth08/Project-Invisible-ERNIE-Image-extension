from logging import getLogger

from modules import script_callbacks

logger = getLogger(__name__)

try:
    from pi_ernie_image.registration import register, unregister

    register()
    script_callbacks.on_script_unloaded(unregister)
except Exception:
    logger.exception(
        "Project Invisible ERNIE Image could not start. "
        "Update the extension and Forge Neo, then restart. ERNIE generation is unavailable."
    )
