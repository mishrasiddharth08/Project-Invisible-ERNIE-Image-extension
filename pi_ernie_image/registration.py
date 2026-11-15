"""Use the extracted ERNIE implementation through Forge's existing dispatch."""

from hashlib import sha256
from importlib import import_module
from pathlib import Path


_HOST_SOURCES = {
    "backend.diffusion_engine.ernie": "c2e72b3e3ec6c500cf95c17deafcc6a1d210cefc7994dad985db613213979b69",
    "backend.text_processing.ernie_engine": "d1fa8cb87bb6801d107f245a5dfb59ecd87b6719c267501a0be7af7abda72688",
    "backend.nn.ernie": "9d6d851145132885358edddfc06683f7179e6e2e4c992416305938c3f85d79d1",
}


def register() -> bool:
    from backend import loader

    if any(cls.__module__ == "pi_ernie_image.engine" for cls in loader.possible_models):
        return True

    host_engine, host_conditioning, host_transformer = (
        import_module(name) for name in _HOST_SOURCES
    )
    for module, expected in zip(
        (host_engine, host_conditioning, host_transformer), _HOST_SOURCES.values()
    ):
        source = Path(module.__file__).read_text(encoding="utf-8")
        if sha256(source.encode()).hexdigest() != expected:
            return False

    original = host_engine.ErnieImage
    if loader.possible_models.count(original) != 1:
        return False
    if host_transformer.ErnieImageModel.__module__ != host_transformer.__name__:
        return False

    from .engine import ErnieImage
    from .transformer import ErnieImageModel

    engines = tuple(ErnieImage if cls is original else cls for cls in loader.possible_models)
    # Forge's transformer factory imports this symbol when loading an ERNIE checkpoint.
    host_transformer.ErnieImageModel = ErnieImageModel
    loader.possible_models = engines
    return True
