from logging import getLogger

from pi_ernie_image.registration import register


try:
    if not register():
        getLogger(__name__).info("Project Invisible ERNIE Image: using the host engine.")
except (AttributeError, ImportError, OSError):
    getLogger(__name__).info("Project Invisible ERNIE Image: host integration unavailable.")
