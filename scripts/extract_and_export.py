#!/usr/bin/env python3
"""
Extract model components and export.

Strategy:
1. Extract ngram_dict (for tokenization in pure Python)
2. Extract embedding weights
3. Export stress_clf MLP
4. Test the full pipeline
"""

import torch
import json
import gzip
import shutil
import numpy as np
from pathlib import Path

torch.set_num_threads(1)

DATA_DIR = Path(__file__).parent.parent / "src" / "silero_stress" / "data"
EXPORT_DIR = DATA_DIR


def load_accentor(lang='ru'):
    """Load accentor with weight restoration."""
    model_name = 'accentor.pt' if lang == 'ru' else f'accentor-{lang}.pt'
    model_path = DATA_DIR / model_name

    with open(model_path, 'rb') as f:
        accentor = torch.package.PackageImporter(f).load_pickle("accentor_models", "accentor")

    if lang == 'ru':
        quantized_weight = accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data.clone()
        restored_weights = accentor.homosolver.model.bert.scale * (quantized_weight - accentor.homosolver.model.bert.zero_point)
        accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data = restored_weights
    else:
        quantized_weight = accentor.accentor.model.embedding.weight.data.clone()
        restored_weights = accentor.accentor.model.scale * (quantized_weight - accentor.accentor.model.zero_point)
        accentor.accentor.model.embedding.weight.data = restored_weights

    return accentor


def extract_ngram_dict(model):
    """Extract ngram_dict from embedding layer."""
    emb = model.embedding
    # The ngram_dict is stored as an attribute in TorchScript
    ngram_dict = dict(emb.ngram_dict)
    return ngram_dict


def extract_exceptions(accentor_obj):
    """Extract exceptions dictionary from accentor."""
    if hasattr(accentor_obj, 'exceptions'):
        exc = accentor_obj.exceptions
        if isinstance(exc, dict):
            # Convert tuple values to lists for JSON serialization
            return {k: list(v) for k, v in exc.items()}
    return {}


def extract_embedding_weights(model):
    """Extract embedding weights with quantization info."""
    emb = model.embedding
    for name, param in emb.named_parameters():
        if 'weight' in name:
            weights = param.data.numpy()
            scale = float(model.scale)
            zero_point = float(model.zero_point)
            return weights, scale, zero_point
    return None, None, None


def extract_stress_clf_weights(model):
    """Extract stress classifier weights."""
    clf = model.stress_clf
    weights = {}
    for name, param in clf.named_parameters():
        weights[name] = param.data.numpy()
    return weights


def word_ngrams(word, min_n=1, max_n=None):
    """
    Reimplementation of word_ngrams function from TorchScript.
    Generates character n-grams with boundary markers.
    """
    if max_n is None:
        max_n = len(word) + 3

    # Add boundary markers
    word_with_markers = f"<{word}>"
    ngrams = []

    for n in range(min_n, max_n + 1):
        for i in range(len(word_with_markers) - n + 1):
            ngram = word_with_markers[i:i+n]
            ngrams.append(ngram)

    return ngrams


def export_ukr_bel_model(lang):
    """Export Ukrainian or Belarusian model components."""
    print(f"\n{'='*60}")
    print(f"Exporting {lang.upper()} model")
    print('='*60)

    # Create language-specific directory
    lang_dir = EXPORT_DIR / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Load RAW model first (before dequantization) to extract int8 weights
    model_path = DATA_DIR / f'accentor-{lang}.pt'
    with open(model_path, 'rb') as f:
        raw_accentor = torch.package.PackageImporter(f).load_pickle("accentor_models", "accentor")
    raw_model = raw_accentor.accentor.model

    # Extract int8 weights BEFORE dequantization
    print("Extracting embedding weights (raw int8)...")
    emb_weights, scale, zero_point = extract_embedding_weights(raw_model)
    print(f"  Shape: {emb_weights.shape}, dtype: {emb_weights.dtype}")
    print(f"  Scale: {scale}, Zero point: {zero_point}")

    emb_path = lang_dir / f"embedding_{lang}.npz"
    np.savez_compressed(
        emb_path,
        weights=emb_weights,
        scale=np.array([scale], dtype=np.float32),
        zero_point=np.array([zero_point], dtype=np.float32)
    )
    print(f"  Saved to {emb_path}")

    # Now load with dequantization for other extractions and testing
    accentor = load_accentor(lang)
    acc_obj = accentor.accentor
    model = acc_obj.model

    # 0. Extract exceptions dictionary
    print("Extracting exceptions...")
    exceptions = extract_exceptions(acc_obj)
    print(f"  Found {len(exceptions)} exceptions")

    exc_path = lang_dir / f"exceptions_{lang}.json"
    with open(exc_path, 'w', encoding='utf-8') as f:
        json.dump(exceptions, f, ensure_ascii=False)
    print(f"  Saved to {exc_path}")

    # 1. Extract ngram_dict
    print("Extracting ngram_dict...")
    ngram_dict = extract_ngram_dict(model)
    print(f"  Found {len(ngram_dict)} n-grams")

    # Save ngram_dict as JSON
    ngram_path = lang_dir / f"ngram_dict_{lang}.json"
    with open(ngram_path, 'w', encoding='utf-8') as f:
        json.dump(ngram_dict, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {ngram_path}")

    # 2. Extract stress_clf MLP weights
    print("Extracting stress classifier...")
    clf_weights = extract_stress_clf_weights(model)
    for name, w in clf_weights.items():
        print(f"  {name}: {w.shape}")

    # Save classifier weights as numpy for pure numpy implementation
    # Note: We use NPZ weights directly in _impl.py
    clf_np_path = lang_dir / f"stress_clf_{lang}_weights.npz"
    np.savez(clf_np_path, **{name.replace('.', '_'): w for name, w in clf_weights.items()})
    print(f"  Saved numpy weights to {clf_np_path}")

    # Return for testing
    return {
        'ngram_dict': ngram_dict,
        'emb_weights': emb_weights,
        'clf_weights': clf_weights,
        'clf_path': clf_np_path,
        'torch_model': model,
        'accentor': accentor.accentor,
    }


def test_numpy_vs_pytorch(exported, lang):
    """Test NumPy implementation against PyTorch."""
    print(f"\n--- Testing NumPy vs PyTorch for {lang} ---")

    # Test words
    test_words = ['привіт', 'читати', 'кнігі'] if lang == 'ukr' else ['кнігі', 'чытаць', 'вечарам']

    ngram_dict = exported['ngram_dict']
    emb_weights = exported['emb_weights']
    clf_weights = exported['clf_weights']
    unk_idx = ngram_dict.get('UNK', 0)

    # Dequantize embeddings
    scale = float(emb_weights.max() - emb_weights.min()) / 255.0 if emb_weights.dtype == np.int8 else 1.0
    # For int8 weights, we need to dequantize
    if emb_weights.dtype == np.int8:
        emb_data = np.load(str(exported['clf_path']).replace('stress_clf', 'embedding').replace('_weights', ''))
        emb_float = emb_data['scale'][0] * (emb_weights.astype(np.float32) - emb_data['zero_point'][0])
    else:
        emb_float = emb_weights.astype(np.float32)

    for word in test_words:
        # Pure Python n-gram extraction
        ngrams = word_ngrams(word, 1, len(word) + 3)
        indices = [ngram_dict.get(ng, unk_idx) for ng in ngrams if ng in ngram_dict]
        if not indices:
            indices = [unk_idx]

        # Compute embedding (mean pooling)
        selected_embs = emb_float[indices]
        embedding = np.mean(selected_embs, axis=0).astype(np.float32)

        # NumPy MLP inference
        x = embedding
        x = np.dot(x, clf_weights['0.weight'].T) + clf_weights['0.bias']
        x = np.maximum(x, 0)  # ReLU
        x = np.dot(x, clf_weights['2.weight'].T) + clf_weights['2.bias']
        x = np.maximum(x, 0)
        x = np.dot(x, clf_weights['4.weight'].T) + clf_weights['4.bias']
        x = np.maximum(x, 0)
        numpy_output = np.dot(x, clf_weights['6.weight'].T) + clf_weights['6.bias']

        # PyTorch inference
        torch_model = exported['torch_model']
        torch_model.eval()
        with torch.no_grad():
            torch_output = torch_model([word])[0].numpy()

        # Compare
        diff = np.abs(numpy_output - torch_output).max()
        print(f"  {word}: max diff = {diff:.6f}")

        # Show predictions
        numpy_pred = np.argmax(numpy_output)
        torch_pred = np.argmax(torch_output)
        print(f"    NumPy pred: {numpy_pred}, PyTorch pred: {torch_pred}")


def gzip_files(lang_dir, remove_originals=True):
    """Gzip JSON files only. NPZ files use internal zip compression."""
    # Only gzip JSON - NPZ already uses zip compression internally
    extensions = ['.json']

    for ext in extensions:
        for filepath in lang_dir.glob(f'*{ext}'):
            gz_path = Path(f'{filepath}.gz')
            # Remove old gzipped file if exists
            if gz_path.exists():
                gz_path.unlink()

            print(f"  Compressing {filepath.name}...")
            with open(filepath, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Report sizes
            orig_size = filepath.stat().st_size / 1024
            gz_size = gz_path.stat().st_size / 1024
            ratio = (1 - gz_size / orig_size) * 100 if orig_size > 0 else 0
            print(f"    {orig_size:.1f} KB -> {gz_size:.1f} KB ({ratio:.1f}% reduction)")

            # Remove original file after successful compression
            if remove_originals:
                filepath.unlink()


def finalize_export(lang):
    """Gzip all files for the language."""
    lang_dir = EXPORT_DIR / lang
    if not lang_dir.exists():
        return

    print(f"\n--- Finalizing {lang.upper()} export ---")
    print("Compressing files...")
    gzip_files(lang_dir)


def main():
    # Export Ukrainian model
    ukr_exported = export_ukr_bel_model('ukr')
    test_numpy_vs_pytorch(ukr_exported, 'ukr')

    # Export Belarusian model
    bel_exported = export_ukr_bel_model('bel')
    test_numpy_vs_pytorch(bel_exported, 'bel')

    # Finalize exports (gzip)
    finalize_export('ukr')
    finalize_export('bel')

    print("\n" + "=" * 60)
    print("UKR/BEL Export Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
