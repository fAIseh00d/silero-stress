#!/usr/bin/env python3
"""
Export Russian silero-stress model to ONNX.

Components:
1. Accentor (ngram) - stress + yo classifiers
2. Homosolver (BERT) - homograph disambiguation
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


def load_ru_accentor():
    """Load Russian accentor with weight restoration."""
    model_path = DATA_DIR / 'accentor.pt'
    with open(model_path, 'rb') as f:
        accentor = torch.package.PackageImporter(f).load_pickle("accentor_models", "accentor")

    # Restore quantized BERT weights
    quantized_weight = accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data.clone()
    restored_weights = accentor.homosolver.model.bert.scale * (quantized_weight - accentor.homosolver.model.bert.zero_point)
    accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data = restored_weights

    return accentor


def extract_ngram_dict(model):
    """Extract ngram_dict from embedding layer."""
    emb = model.embedding
    return dict(emb.ngram_dict)


def extract_embedding_weights(model):
    """Extract embedding weights (raw values stored as float32).

    Note: The model stores quantized-like values as float32. These should
    be used as-is during inference - the model does NOT apply scale/zero_point
    to the n-gram embeddings during forward pass.
    """
    emb = model.embedding
    scale = float(model.scale)
    zero_point = float(model.zero_point)
    for name, param in emb.named_parameters():
        if 'weight' in name:
            # Export raw weights exactly as stored in model (NOT dequantized)
            return param.data.numpy(), scale, zero_point
    return None, None, None


def extract_clf_weights(clf, prefix=""):
    """Extract classifier weights from a Sequential module."""
    weights = {}
    for name, param in clf.named_parameters():
        weights[f"{prefix}{name}"] = param.data.numpy()
    return weights


def export_accentor_component(accentor):
    """Export the accentor (ngram) component."""
    print("\n" + "="*60)
    print("Exporting RU ACCENTOR (ngram) component")
    print("="*60)

    # Create ru subdirectory
    ru_dir = EXPORT_DIR / "ru"
    ru_dir.mkdir(parents=True, exist_ok=True)

    acc = accentor.accentor
    model = acc.model

    # 1. Extract exceptions
    print("Extracting exceptions...")
    if hasattr(acc, 'exceptions') and isinstance(acc.exceptions, dict):
        exceptions = {k: list(v) for k, v in acc.exceptions.items()}
    else:
        exceptions = {}
    print(f"  Found {len(exceptions)} exceptions")

    exc_path = ru_dir / "exceptions_ru.json"
    with open(exc_path, 'w', encoding='utf-8') as f:
        json.dump(exceptions, f, ensure_ascii=False)
    print(f"  Saved to {exc_path}")

    # 2. Extract ngram_dict
    print("Extracting ngram_dict...")
    ngram_dict = extract_ngram_dict(model)
    print(f"  Found {len(ngram_dict)} n-grams")

    ngram_path = ru_dir / "ngram_dict_ru.json"
    with open(ngram_path, 'w', encoding='utf-8') as f:
        json.dump(ngram_dict, f, ensure_ascii=False)
    print(f"  Saved to {ngram_path}")

    # 3. Extract embedding weights
    print("Extracting embedding weights...")
    emb_weights, scale, zero_point = extract_embedding_weights(model)
    print(f"  Shape: {emb_weights.shape}, dtype: {emb_weights.dtype}")
    print(f"  Scale: {scale}, Zero point: {zero_point}")

    emb_path = ru_dir / "embedding_ru.npz"
    np.savez_compressed(
        emb_path,
        weights=emb_weights,
        scale=np.array([scale], dtype=np.float32),
        zero_point=np.array([zero_point], dtype=np.float32)
    )
    print(f"  Saved to {emb_path}")

    # 4. Extract stress classifier weights
    print("Extracting stress classifier...")
    stress_clf_weights = extract_clf_weights(model.stress_clf)
    for name, w in stress_clf_weights.items():
        print(f"  {name}: {w.shape}")

    stress_clf_path = ru_dir / "stress_clf_ru_weights.npz"
    np.savez(stress_clf_path, **{n.replace('.', '_'): w for n, w in stress_clf_weights.items()})
    print(f"  Saved to {stress_clf_path}")

    # 5. Extract yo classifier weights
    print("Extracting yo classifier...")
    yo_clf_weights = extract_clf_weights(model.yo_clf)
    for name, w in yo_clf_weights.items():
        print(f"  {name}: {w.shape}")

    yo_clf_path = ru_dir / "yo_clf_ru_weights.npz"
    np.savez(yo_clf_path, **{n.replace('.', '_'): w for n, w in yo_clf_weights.items()})
    print(f"  Saved to {yo_clf_path}")

    # Note: stress_clf and yo_clf use NPZ weights (not ONNX) for inference
    # See _impl.py - it loads stress_clf_ru_weights.npz.gz and yo_clf_ru_weights.npz.gz

    return {
        'ngram_dict': ngram_dict,
        'emb_weights': emb_weights,
        'stress_clf_weights': stress_clf_weights,
        'yo_clf_weights': yo_clf_weights,
        'exceptions': exceptions,
    }


def export_homosolver_component(accentor):
    """Export the homosolver (BERT) component."""
    print("\n" + "="*60)
    print("Exporting RU HOMOSOLVER (BERT) component")
    print("="*60)

    ru_dir = EXPORT_DIR / "ru"
    ru_dir.mkdir(parents=True, exist_ok=True)

    hs = accentor.homosolver
    model = hs.model

    # 1. Export homodict
    print("Extracting homodict...")
    homodict = dict(hs.homodict)
    print(f"  Found {len(homodict)} homographs")

    homodict_path = ru_dir / "homodict_ru.json"
    with open(homodict_path, 'w', encoding='utf-8') as f:
        json.dump(homodict, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {homodict_path}")

    # 2. Export yohomodict
    print("Extracting yohomodict...")
    yohomodict = dict(hs.yohomodict)
    print(f"  Found {len(yohomodict)} yo-homographs")

    yohomodict_path = ru_dir / "yohomodict_ru.json"
    with open(yohomodict_path, 'w', encoding='utf-8') as f:
        json.dump(yohomodict, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {yohomodict_path}")

    # 3. Export tokenizer vocab
    print("Extracting tokenizer vocab...")
    tok = hs.tokenizer
    vocab = dict(tok.vocab)
    print(f"  Vocab size: {len(vocab)}")

    vocab_path = ru_dir / "vocab_ru.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  Saved to {vocab_path}")

    # Export special tokens
    special_tokens = {
        'pad_token_id': tok.pad_token_id,
        'cls_token_id': tok.cls_token_id,
        'sep_token_id': tok.sep_token_id,
        'unk_token_id': tok.unk_token_id,
        'homo_start_id': tok.homo_start_id,
        'homo_end_id': tok.homo_end_id,
    }
    print(f"  Special tokens: {special_tokens}")

    special_path = ru_dir / "special_tokens_ru.json"
    with open(special_path, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f)
    print(f"  Saved to {special_path}")

    # Note: BERT and homo_clf ONNX export is handled by export_bert_direct.py
    # This script exports dictionaries and tokenizer data only

    return {
        'homodict': homodict,
        'yohomodict': yohomodict,
        'vocab': vocab,
        'special_tokens': special_tokens,
    }


def merge_onnx_external_data(lang_dir):
    """Convert ONNX files with external .data files to embedded format."""
    try:
        import onnx
    except ImportError:
        print("  onnx not installed, skipping merge")
        return

    for onnx_path in lang_dir.glob('*.onnx'):
        data_path = Path(f'{onnx_path}.data')
        if data_path.exists():
            print(f"  Merging {onnx_path.name} external data...")
            model = onnx.load(str(onnx_path), load_external_data=True)
            onnx.save(model, str(onnx_path), save_as_external_data=False)
            data_path.unlink()
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            print(f"    Done. Size: {size_mb:.1f} MB")


def gzip_files(lang_dir, remove_originals=True):
    """Gzip JSON files only. NPZ files use internal zip compression."""
    # Only gzip JSON - NPZ already uses zip compression internally
    # Note: ONNX files are handled by export_bert_direct.py
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


def finalize_export():
    """Merge ONNX external data and gzip all files for Russian."""
    ru_dir = EXPORT_DIR / 'ru'
    if not ru_dir.exists():
        return

    print("\n--- Finalizing RU export ---")

    print("Merging ONNX external data...")
    merge_onnx_external_data(ru_dir)

    print("Compressing files...")
    gzip_files(ru_dir)


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    accentor = load_ru_accentor()

    # Export accentor component
    export_accentor_component(accentor)

    # Export homosolver component
    export_homosolver_component(accentor)

    # Finalize (merge ONNX, gzip)
    finalize_export()

    print("\n" + "="*60)
    print("RU EXPORT COMPLETE")
    print("="*60)
    print(f"Files exported to: {EXPORT_DIR / 'ru'}")


if __name__ == "__main__":
    main()
