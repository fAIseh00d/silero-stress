#!/usr/bin/env python3
"""
Benchmark silero-stress across all available backends and devices.

Auto-detects available ONNX providers and PyTorch CUDA support.

Usage:
    PYTHONPATH=src python scripts/benchmark.py
    PYTHONPATH=src python scripts/benchmark.py --onnx-only
    PYTHONPATH=src python scripts/benchmark.py --torch-only
"""

import argparse
import gc
import time
import sys

# Test data
TEST_SENTENCES_100 = [
    "Привет, как дела?",
    "Москва столица России.",
    "Меня зовут Лева Королев.",
    "Я уже готов открыть все ваши замки любой сложности!",
    "Древний замок стоит на горе.",
] * 20  # 100 sentences

# Sentence with 7 homographs for BERT-heavy benchmark
HOMO_SENTENCE = 'Он готов идти дома, потом стоит там, но тому человеку уже надо уже было идти.'

N_WARMUP = 10
N_RUNS = 50


def detect_onnx_devices():
    """Detect available ONNX execution providers and map to device names."""
    try:
        import onnxruntime as ort
    except ImportError:
        return []

    provider_map = {
        'CPUExecutionProvider': 'cpu',
        'CUDAExecutionProvider': 'cuda',
        'TensorrtExecutionProvider': 'tensorrt',
        'ROCMExecutionProvider': 'rocm',
        'OpenVINOExecutionProvider': 'openvino',
        'CoreMLExecutionProvider': 'coreml',
        'DmlExecutionProvider': 'directml',
    }

    # Order: CPU first (baseline), then accelerators
    preferred_order = ['cpu', 'cuda', 'rocm', 'tensorrt', 'openvino', 'coreml', 'directml']

    available = ort.get_available_providers()
    detected = set()
    for provider in available:
        if provider in provider_map:
            detected.add(provider_map[provider])

    # Return in preferred order
    return [d for d in preferred_order if d in detected]


def detect_torch_devices():
    """Detect available PyTorch devices."""
    try:
        import torch
    except ImportError:
        return []

    devices = ['cpu']

    if torch.cuda.is_available():
        devices.append('cuda')

    # ROCm uses the same cuda API in PyTorch
    # MPS for Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')

    return devices


def benchmark_backend(backend, device, sentences, homo_sentence):
    """Benchmark a single backend+device combination.

    Args:
        backend: 'torch' or 'onnx'
        device: device string (cpu, cuda, tensorrt, etc.)
        sentences: list of sentences for throughput test
        homo_sentence: single sentence with many homographs

    Returns:
        dict with timing results
    """
    # Load model
    load_start = time.perf_counter()

    if backend == 'torch':
        from silero_stress._torch_impl import TorchAccentorRu
        acc = TorchAccentorRu(device=device)
    else:
        from silero_stress._impl import AccentorRu
        acc = AccentorRu(device=device)

    load_time = (time.perf_counter() - load_start) * 1000

    # Warmup
    for s in sentences[:N_WARMUP]:
        acc(s)
    acc(homo_sentence)

    # Benchmark: 100 sentences
    gc.collect()
    start = time.perf_counter()
    for _ in range(N_RUNS):
        for s in sentences:
            acc(s)
    sentences_time = (time.perf_counter() - start) * 1000 / N_RUNS

    # Benchmark: homograph-heavy sentence
    gc.collect()
    start = time.perf_counter()
    for _ in range(N_RUNS * 2):
        acc(homo_sentence)
    homo_time = (time.perf_counter() - start) * 1000 / (N_RUNS * 2)

    return {
        'load_ms': load_time,
        'sentences_100_ms': sentences_time,
        'homo_7_ms': homo_time,
    }


def print_results(results, baseline_key=None):
    """Print benchmark results table."""
    if not results:
        print("No results to display.")
        return

    # Find baseline for speedup calculation
    if baseline_key is None:
        # Prefer onnx_cpu, fallback to torch_cpu, then first available
        for key in ['onnx_cpu', 'torch_cpu']:
            if key in results:
                baseline_key = key
                break
        if baseline_key is None:
            baseline_key = list(results.keys())[0]

    baseline = results.get(baseline_key, {})
    baseline_sent = baseline.get('sentences_100_ms', 1)
    baseline_homo = baseline.get('homo_7_ms', 1)

    print(f"\n| Backend        | Load    | 100 sentences    | 1 sent (7 homo)  |")
    print(f"|----------------|---------|------------------|------------------|")

    for name, data in results.items():
        sent_speedup = baseline_sent / data['sentences_100_ms'] if data['sentences_100_ms'] > 0 else 0
        homo_speedup = baseline_homo / data['homo_7_ms'] if data['homo_7_ms'] > 0 else 0

        if name == baseline_key:
            print(f"| {name:14s} | {data['load_ms']:5.0f}ms | {data['sentences_100_ms']:6.1f}ms          | {data['homo_7_ms']:6.2f}ms          |")
        else:
            print(f"| {name:14s} | {data['load_ms']:5.0f}ms | {data['sentences_100_ms']:6.1f}ms ({sent_speedup:4.1f}x)  | {data['homo_7_ms']:6.2f}ms ({homo_speedup:4.1f}x)  |")


def main():
    parser = argparse.ArgumentParser(description='Benchmark silero-stress')
    parser.add_argument('--onnx-only', action='store_true', help='Only benchmark ONNX backends')
    parser.add_argument('--torch-only', action='store_true', help='Only benchmark PyTorch backends')
    parser.add_argument('--devices', '-d', default=None,
                        help='Override auto-detection with specific devices (e.g., cpu,cuda)')
    args = parser.parse_args()

    print("=" * 70)
    print("SILERO-STRESS BENCHMARK")
    print("=" * 70)

    # Detect available backends
    onnx_devices = [] if args.torch_only else detect_onnx_devices()
    torch_devices = [] if args.onnx_only else detect_torch_devices()

    # Override with manual device list if specified
    if args.devices:
        manual_devices = [d.strip().lower() for d in args.devices.split(',')]
        if not args.torch_only:
            onnx_devices = [d for d in manual_devices if d in detect_onnx_devices()]
        if not args.onnx_only:
            torch_devices = [d for d in manual_devices if d in detect_torch_devices()]

    # Print detected backends
    print(f"\nONNX devices:   {onnx_devices if onnx_devices else 'none'}")
    print(f"PyTorch devices: {torch_devices if torch_devices else 'none'}")

    if not onnx_devices and not torch_devices:
        print("\nNo backends available!")
        return 1

    # Print versions
    try:
        import onnxruntime as ort
        print(f"\nONNX Runtime: {ort.__version__}")
        print(f"  Providers: {ort.get_available_providers()}")
    except ImportError:
        pass

    try:
        import torch
        torch.set_num_threads(1)
        print(f"PyTorch: {torch.__version__} (threads=1)")
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    import numpy as np
    print(f"NumPy: {np.__version__}")

    # Run benchmarks
    results = {}

    print(f"\nBenchmarking (warmup={N_WARMUP}, runs={N_RUNS})...")

    for device in torch_devices:
        name = f"torch_{device}"
        print(f"  {name}...", end=" ", flush=True)
        try:
            results[name] = benchmark_backend('torch', device, TEST_SENTENCES_100, HOMO_SENTENCE)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    for device in onnx_devices:
        name = f"onnx_{device}"
        print(f"  {name}...", end=" ", flush=True)
        try:
            results[name] = benchmark_backend('onnx', device, TEST_SENTENCES_100, HOMO_SENTENCE)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print_results(results)

    # Summary
    if 'torch_cpu' in results and 'onnx_cpu' in results:
        speedup = results['torch_cpu']['sentences_100_ms'] / results['onnx_cpu']['sentences_100_ms']
        print(f"\nONNX CPU is {speedup:.1f}x faster than PyTorch CPU")

    if 'onnx_cpu' in results and 'onnx_tensorrt' in results:
        speedup = results['onnx_cpu']['sentences_100_ms'] / results['onnx_tensorrt']['sentences_100_ms']
        print(f"ONNX TensorRT is {speedup:.1f}x faster than ONNX CPU")

    return 0


if __name__ == '__main__':
    sys.exit(main())
