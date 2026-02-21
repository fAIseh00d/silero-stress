#!/usr/bin/env python3
"""
Export BERT model directly from TorchScript (no transformers dependency).
This ensures 100% match with PyTorch predictions.
"""

import torch
import gzip
import shutil
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "src" / "silero_stress" / "data"
EXPORT_DIR = DATA_DIR / "ru"


def merge_onnx_external_data():
    """Convert ONNX files with external .data files to embedded format."""
    try:
        import onnx
    except ImportError:
        print("  onnx not installed, skipping merge")
        return

    for onnx_path in EXPORT_DIR.glob('*.onnx'):
        data_path = Path(f'{onnx_path}.data')
        if data_path.exists():
            print(f"  Merging {onnx_path.name} external data...")
            model = onnx.load(str(onnx_path), load_external_data=True)
            onnx.save(model, str(onnx_path), save_as_external_data=False)
            data_path.unlink()
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            print(f"    Done. Size: {size_mb:.1f} MB")


def gzip_files(remove_originals=True):
    """Gzip all data files created by this script."""
    # Only gzip ONNX files - NPZ files from export_weights() are debug-only
    extensions = ['*.onnx']

    for pattern in extensions:
        for filepath in EXPORT_DIR.glob(pattern):
            gz_path = Path(f'{filepath}.gz')
            if gz_path.exists():
                gz_path.unlink()

            print(f"  Compressing {filepath.name}...")
            with open(filepath, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            orig_size = filepath.stat().st_size / 1024 / 1024
            gz_size = gz_path.stat().st_size / 1024 / 1024
            ratio = (1 - gz_size / orig_size) * 100 if orig_size > 0 else 0
            print(f"    {orig_size:.1f} MB -> {gz_size:.1f} MB ({ratio:.1f}% reduction)")

            # Remove original file after successful compression
            if remove_originals:
                filepath.unlink()


def load_model():
    """Load the PyTorch model."""
    model_path = DATA_DIR / 'accentor.pt'
    importer = torch.package.PackageImporter(str(model_path))
    return importer.load_pickle('accentor_models', 'accentor')


def export_weights():
    """Export all weights as NPZ files (for debugging only - not used in inference)."""
    acc = load_model()
    bert = acc.homosolver.model.bert
    clf = acc.homosolver.model

    print("Exporting BERT weights (DEBUG ONLY)...")

    bert_weights = {}
    for name, param in bert.named_parameters():
        data = param.data
        if data.dtype == torch.int8:
            # Cast int8 to float32 (NO dequantization - this is how PyTorch uses them)
            data = data.float()
        bert_weights[name] = data.numpy()
        print(f"  {name}: {data.shape}")

    np.savez(EXPORT_DIR / 'bert_weights.npz', **bert_weights)
    print(f"Saved to {EXPORT_DIR / 'bert_weights.npz'}")
    print("  NOTE: This file is NOT used in inference - BERT is exported to ONNX")

    print("\nExporting classifier weights (DEBUG ONLY)...")
    clf_weights = {}
    for name, param in clf.named_parameters():
        if 'bert' not in name:
            clf_weights[name] = param.data.numpy()
            print(f"  {name}: {param.shape}")

    np.savez(EXPORT_DIR / 'homo_clf_weights.npz', **clf_weights)
    print(f"Saved to {EXPORT_DIR / 'homo_clf_weights.npz'}")
    print("  NOTE: This file is NOT used in inference - homo_clf is exported to ONNX")


def export_onnx():
    """Export BERT and classifier to ONNX format using transformers."""
    from transformers import BertConfig, BertModel

    acc = load_model()
    bert = acc.homosolver.model.bert

    # Dequantize BERT embeddings in place (like TorchAccentorRu does)
    scale = float(bert.scale)
    zero_point = float(bert.zero_point)
    qw = bert.embeddings.word_embeddings.weight.data.clone()
    bert.embeddings.word_embeddings.weight.data = scale * (qw.float() - zero_point)

    print("\nExporting BERT to ONNX...")

    # Extract configuration
    config = BertConfig(
        vocab_size=83830,
        hidden_size=312,
        num_hidden_layers=3,
        num_attention_heads=12,
        intermediate_size=600,
        max_position_embeddings=2048,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,  # No dropout in inference
        attention_probs_dropout_prob=0.0,
    )

    # Create transformers model
    tf_bert = BertModel(config)

    # Copy weights from TorchScript
    state_dict = {}

    # Get dequantization parameters
    scale = float(bert.scale)
    zero_point = float(bert.zero_point)

    for name, param in bert.named_parameters():
        data = param.data
        if data.dtype == torch.int8:
            # Dequantize int8 to float32 (matches TorchAccentorRu behavior)
            data = scale * (data.float() - zero_point)

        # Map names to transformers format
        tf_name = name
        state_dict[tf_name] = data

    # Load into transformers model
    tf_bert.load_state_dict(state_dict, strict=False)
    tf_bert.eval()

    # Test with same input
    dummy_input = torch.tensor([[2, 3280, 83828, 29600, 83829, 18, 3]], dtype=torch.int64)

    with torch.no_grad():
        # Original PyTorch model
        pt_out = bert(dummy_input)[0]

        # Transformers model
        tf_out = tf_bert(dummy_input).last_hidden_state

    print(f"PyTorch output[:5]: {pt_out[0, 3, :5].tolist()}")
    print(f"Transformers output[:5]: {tf_out[0, 3, :5].tolist()}")
    print(f"Max diff: {(pt_out - tf_out).abs().max():.6f}")

    # Export to ONNX
    onnx_path = EXPORT_DIR / 'bert_ru.onnx'
    torch.onnx.export(
        tf_bert,
        dummy_input,
        str(onnx_path),
        input_names=['input_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'hidden_states': {0: 'batch', 1: 'seq_len'}
        },
        opset_version=14
    )
    print(f"Saved to {onnx_path}")

    # Verify ONNX
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(None, {'input_ids': dummy_input.numpy()})[0]

    diff = np.abs(pt_out.numpy() - onnx_out).max()
    print(f"ONNX vs PyTorch max diff: {diff:.6f}")

    return onnx_path


def export_classifier_onnx():
    """Export the homograph classifier to ONNX."""
    acc = load_model()
    clf = acc.homosolver.model

    print("\nExporting classifier to ONNX...")

    # Get weights from named_parameters (TorchScript doesn't support indexing)
    clf_weights = {}
    for name, param in clf.named_parameters():
        if 'bert' not in name:
            clf_weights[name] = param.data

    # Create a simple classifier module
    class ClassifierWrapper(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.fc1_weight = weights['homo_clf.0.weight']
            self.fc1_bias = weights['homo_clf.0.bias']
            self.fc2_weight = weights['homo_clf.3.weight']
            self.fc2_bias = weights['homo_clf.3.bias']

        def forward(self, x):
            # Linear -> ReLU -> Linear (no dropout in inference)
            x = torch.nn.functional.linear(x, self.fc1_weight, self.fc1_bias)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.linear(x, self.fc2_weight, self.fc2_bias)
            return x

    wrapper = ClassifierWrapper(clf_weights)
    wrapper.eval()

    dummy_input = torch.randn(1, 624)

    onnx_path = EXPORT_DIR / 'homo_clf_ru.onnx'
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=['embedding'],
        output_names=['logit'],
        dynamic_axes={
            'embedding': {0: 'batch'},
            'logit': {0: 'batch'}
        },
        opset_version=14
    )
    print(f"Saved to {onnx_path}")

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))

    with torch.no_grad():
        pt_out = wrapper(dummy_input)

    onnx_out = session.run(None, {'embedding': dummy_input.numpy()})[0]

    diff = np.abs(pt_out.numpy() - onnx_out).max()
    print(f"Max diff vs PyTorch: {diff:.6f}")

    return onnx_path


def test_full_pipeline():
    """Test the full homograph prediction pipeline."""
    import onnxruntime as ort

    acc = load_model()
    tok = acc.homosolver.tokenizer

    bert_path = EXPORT_DIR / 'bert_ru.onnx'
    clf_path = EXPORT_DIR / 'homo_clf_ru.onnx'

    if not bert_path.exists() or not clf_path.exists():
        print("ONNX files not found. Run export first.")
        return

    bert_onnx = ort.InferenceSession(str(bert_path))
    clf_onnx = ort.InferenceSession(str(clf_path))

    # Test homographs
    homodict = acc.homosolver.homodict

    test_words = ['готов', 'дома', 'тому', 'потом', 'стоит']
    matches = 0
    total = 0

    print("\n=== Testing Homograph Predictions ===\n")

    for word in test_words:
        if word not in homodict:
            continue

        sent = f'Он {word}.'
        marked = f'Он [HOMO] {word} [/HOMO] .'

        # PyTorch prediction
        pt_result = acc.homosolver(sent, homodict)

        # ONNX prediction
        ids = tok.encode(marked)
        input_ids = np.array([ids], dtype=np.int64)

        homo_start = ids.index(83828)
        homo_end = ids.index(83829)

        hidden = bert_onnx.run(None, {'input_ids': input_ids})[0]

        # Extract embeddings (marker + word mean)
        marker_emb = hidden[0, homo_start, :]
        word_tokens = hidden[0, homo_start + 1:homo_end, :]
        word_emb = word_tokens.mean(axis=0)
        combined = np.concatenate([marker_emb, word_emb]).reshape(1, -1).astype(np.float32)

        logit = clf_onnx.run(None, {'embedding': combined})[0]
        sigmoid_val = 1 / (1 + np.exp(-logit[0, 0]))
        pred = 1 if sigmoid_val > 0.5 else 0

        variants = sorted(homodict[word])
        onnx_pred = variants[pred]

        # Extract PyTorch prediction (strip punctuation for comparison)
        pt_word = [w for w in pt_result.split() if '+' in w and word in w.lower().replace('+', '')]
        pt_word = pt_word[0] if pt_word else ''
        pt_word_clean = pt_word.rstrip('.,!?;:')  # Remove trailing punctuation

        match = "✓" if pt_word_clean.lower() == onnx_pred.lower() else "✗"
        print(f"{match} {word}: PT={pt_word_clean} ONNX={onnx_pred}")

        if pt_word_clean.lower() == onnx_pred.lower():
            matches += 1
        total += 1

    print(f"\nMatch rate: {matches}/{total} ({100*matches/total:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export BERT from TorchScript")
    parser.add_argument('--weights', action='store_true', help='Export NPZ weights')
    parser.add_argument('--onnx', action='store_true', help='Export ONNX models')
    parser.add_argument('--test', action='store_true', help='Test full pipeline')
    parser.add_argument('--all', action='store_true', help='Run all exports and tests')
    parser.add_argument('--no-finalize', action='store_true', help='Skip ONNX merge and gzip')
    args = parser.parse_args()

    if args.weights:  # --weights is for debugging only, not included in --all
        export_weights()

    if args.all or args.onnx:
        export_onnx()
        export_classifier_onnx()

    if args.all or args.test:
        test_full_pipeline()

    # Finalize: merge ONNX external data and gzip
    if (args.all or args.onnx or args.weights) and not args.no_finalize:
        print("\n--- Finalizing BERT export ---")
        print("Merging ONNX external data...")
        merge_onnx_external_data()
        print("Compressing files...")
        gzip_files()

    if not (args.weights or args.onnx or args.test or args.all):
        parser.print_help()
