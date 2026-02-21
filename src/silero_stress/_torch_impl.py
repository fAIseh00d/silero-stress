"""
PyTorch implementation of silero-stress accentor.
Wraps the original TorchScript models (.pt files).
"""

import torch
torch.set_num_threads(1)

try:
    import importlib_resources as impresources
except ImportError:
    from importlib import resources as impresources


def _get_data_path(filename):
    """Get path to data file using importlib.resources."""
    package_path = "silero_stress.data"
    if hasattr(impresources, 'files'):
        return str(impresources.files(package_path).joinpath(filename))
    else:
        with impresources.path(package_path, filename) as f:
            return str(f)


def _normalize_device_str(device):
    """Normalize device to string format expected by the original model.

    The original model expects: 'cpu', 'cuda:0', 'cuda:1', etc.
    """
    if hasattr(device, 'type'):  # torch.device
        device = str(device)
    device = device.lower()
    if device in ('tensorrt', 'trt'):
        device = 'cuda:0'
    elif device == 'cuda':
        device = 'cuda:0'
    # cuda:0, cuda:1, etc. are already in correct format
    return device


class TorchAccentorRu:
    """PyTorch-based Russian accentor wrapping the original .pt model."""

    def __init__(self, use_homosolver=True, device='cpu'):
        self._use_homosolver = use_homosolver
        self._device_str = _normalize_device_str(device)

        # Load model
        model_path = _get_data_path('accentor.pt')
        importer = torch.package.PackageImporter(model_path)
        self._model = importer.load_pickle('accentor_models', 'accentor')

        # Restore quantized BERT embeddings (int8 -> float32)
        bert = self._model.homosolver.model.bert
        qw = bert.embeddings.word_embeddings.weight.data.clone()
        rw = bert.scale * (qw - bert.zero_point)
        bert.embeddings.word_embeddings.weight.data = rw

        # N-gram embeddings are already float32, do NOT dequantize

        # Move to device
        if self._device_str != 'cpu':
            self._model.to(self._device_str)

    def to(self, device):
        """Move model to device. Returns self for chaining."""
        self._device_str = _normalize_device_str(device)
        self._model.to(self._device_str)
        return self

    @property
    def device(self):
        """Return current device string."""
        return self._device_str

    def __call__(self, sentence, put_stress=True, put_yo=True,
                 put_stress_homo=True, put_yo_homo=True, stress_single_vowel=True, **kwargs):
        """Process sentence and add stress marks.

        Args:
            sentence: Input text
            put_stress: Add stress marks to words
            put_yo: Replace е with ё where appropriate
            put_stress_homo: Add stress marks to homographs (passed to underlying model)
            put_yo_homo: Replace е with ё in homographs (passed to underlying model)
            stress_single_vowel: Add stress even for single-vowel words
        """
        return self._model(
            sentence,
            put_stress=put_stress,
            put_yo=put_yo,
            stress_single_vowel=stress_single_vowel
        )


class TorchAccentorUkrBel:
    """PyTorch-based Ukrainian/Belarusian accentor wrapping the original .pt model."""

    def __init__(self, lang):
        self.lang = lang
        self._device_str = 'cpu'

        # Load model
        model_path = _get_data_path(f'accentor-{lang}.pt')
        importer = torch.package.PackageImporter(model_path)
        self._model = importer.load_pickle('accentor_models', 'accentor')

        # Restore quantized embeddings
        ngram_model = self._model.accentor.model
        qw = ngram_model.embedding.weight.data.clone()
        rw = ngram_model.scale * (qw - ngram_model.zero_point)
        ngram_model.embedding.weight.data = rw

    def to(self, device):
        """Move model to device. Returns self for chaining.

        Note: UKR/BEL models don't have full GPU support in the original,
        but we support .to() for API consistency.
        """
        self._device_str = _normalize_device_str(device)
        return self

    @property
    def device(self):
        """Return current device string."""
        return self._device_str

    def __call__(self, sentence, **kwargs):
        """Process sentence and add stress marks."""
        return self._model(sentence)
