"""
Russian/Ukrainian/Belarusian Word Stress by Silero.

Supports dual backends:
- PyTorch (original implementation) - if torch is installed
- NumPy/ONNX (lightweight) - pure NumPy, ONNX only for Russian homographs

Supports:
- Russian (ru): stress, yo replacement, homograph disambiguation
- Ukrainian (ukr): stress
- Belarusian (bel): stress
"""

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_accentor(lang='ru', use_homosolver=True, device='cpu', backend=None):
    """
    Load an accentor for the specified language.

    Args:
        lang: Language code ('ru', 'ukr', or 'bel')
        use_homosolver: For Russian, whether to use BERT for homograph disambiguation
                       (requires onnxruntime for numpy backend)
        device: Device for inference: 'cpu', 'cuda', 'cuda:0', or 'tensorrt'
        backend: 'torch', 'numpy', or None (auto-detect, prefers torch)

    Returns:
        Accentor object with __call__ and .to() methods

    Example:
        >>> accentor = load_accentor('ru')
        >>> accentor("Привет мир")
        'Прив+ет м+ир'

        >>> # Move to GPU
        >>> accentor.to('cuda:0')
        >>> accentor("Привет мир")
        'Прив+ет м+ир'

        >>> # Force specific backend
        >>> accentor = load_accentor('ru', backend='numpy')
    """
    # Auto-detect backend: prefer torch if available
    if backend is None:
        backend = 'torch' if HAS_TORCH else 'numpy'
    elif backend == 'torch' and not HAS_TORCH:
        raise ImportError(
            "PyTorch not installed. Install with: pip install silero-stress[torch]\n"
            "Or use backend='numpy' for the lightweight implementation."
        )

    if backend == 'torch':
        from silero_stress._torch_impl import TorchAccentorRu, TorchAccentorUkrBel

        if lang == 'ru':
            return TorchAccentorRu(use_homosolver=use_homosolver, device=device)
        elif lang in ('ukr', 'bel'):
            acc = TorchAccentorUkrBel(lang)
            if device != 'cpu':
                acc.to(device)
            return acc
        else:
            raise ValueError(f'Unsupported language {lang}. Must be in ["ru", "ukr", "bel"]')

    else:  # numpy backend
        from silero_stress._impl import AccentorRu, AccentorUkrBel

        if lang == 'ru':
            return AccentorRu(use_homosolver=use_homosolver, device=device)
        elif lang in ('ukr', 'bel'):
            return AccentorUkrBel(lang)
        else:
            raise ValueError(f'Unsupported language {lang}. Must be in ["ru", "ukr", "bel"]')
