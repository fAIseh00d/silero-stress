#!/usr/bin/env python3
"""
Benchmark: compare PyTorch vs NumPy/ONNX performance.

Usage:
    # Benchmark both backends (requires torch venv):
    PYTHONPATH=src .venv/bin/python scripts/benchmark.py

    # Benchmark numpy only (works in onnx_venv):
    PYTHONPATH=src .onnx_venv/bin/python scripts/benchmark.py --numpy-only
"""

import argparse
import gc
import time
import statistics
import sys

# Test sentences
TEST_SENTENCES = {
    'ru': [
        "Привет, как дела?",
        "Москва столица России.",
        "Меня зовут Лева Королев.",
        "Я уже готов открыть все ваши замки любой сложности!",
        "Древний замок стоит на горе.",
    ],
    'ukr': [
        "Привіт, як справи?",
        "Київ столиця України.",
        "Добрий день, як ваші справи?",
        "Львів дуже гарне місто.",
        "Дякую за допомогу.",
    ],
    'bel': [
        "Добры дзень, як справы?",
        "Мінск сталіца Беларусі.",
        "Дзякуй вельмі за дапамогу.",
        "Гродна старажытны горад.",
        "Віцебск на Дзвіне.",
    ],
}

WARMUP_ITERATIONS = 50
BENCHMARK_ITERATIONS = 500


def benchmark_torch(lang, use_homosolver=True):
    """Benchmark PyTorch implementation."""
    from silero_stress._torch_impl import TorchAccentorRu, TorchAccentorUkrBel

    # Load
    load_start = time.perf_counter()
    if lang == 'ru':
        acc = TorchAccentorRu(use_homosolver=use_homosolver, device='cpu')
    else:
        acc = TorchAccentorUkrBel(lang)
    load_time = time.perf_counter() - load_start

    sentences = TEST_SENTENCES[lang]

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        for sent in sentences:
            acc(sent)

    # Benchmark
    gc.collect()
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        for sent in sentences:
            acc(sent)
        times.append(time.perf_counter() - start)

    return {
        'load_time_ms': load_time * 1000,
        'batch_mean_ms': statistics.mean(times) * 1000,
        'batch_std_ms': statistics.stdev(times) * 1000,
        'per_sentence_ms': statistics.mean(times) * 1000 / len(sentences),
        'p50_ms': statistics.median(times) * 1000,
        'p95_ms': sorted(times)[int(0.95 * len(times))] * 1000,
        'p99_ms': sorted(times)[int(0.99 * len(times))] * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
    }


def benchmark_numpy(lang, use_homosolver=True):
    """Benchmark NumPy/ONNX implementation."""
    from silero_stress._impl import AccentorRu, AccentorUkrBel

    # Load
    load_start = time.perf_counter()
    if lang == 'ru':
        acc = AccentorRu(use_homosolver=use_homosolver, device='cpu')
    else:
        acc = AccentorUkrBel(lang)
    load_time = time.perf_counter() - load_start

    sentences = TEST_SENTENCES[lang]

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        for sent in sentences:
            acc(sent)

    # Benchmark
    gc.collect()
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        for sent in sentences:
            acc(sent)
        times.append(time.perf_counter() - start)

    return {
        'load_time_ms': load_time * 1000,
        'batch_mean_ms': statistics.mean(times) * 1000,
        'batch_std_ms': statistics.stdev(times) * 1000,
        'per_sentence_ms': statistics.mean(times) * 1000 / len(sentences),
        'p50_ms': statistics.median(times) * 1000,
        'p95_ms': sorted(times)[int(0.95 * len(times))] * 1000,
        'p99_ms': sorted(times)[int(0.99 * len(times))] * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
    }


def print_results(lang, torch_results, numpy_results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {lang.upper()}")
    print(f"{'='*70}")
    print(f"Sentences per batch: {len(TEST_SENTENCES[lang])}")
    print(f"Warmup: {WARMUP_ITERATIONS}, Iterations: {BENCHMARK_ITERATIONS}")

    if torch_results:
        print(f"\n[PyTorch]")
        print(f"  Load time:      {torch_results['load_time_ms']:8.1f}ms")
        print(f"  Per batch:      {torch_results['batch_mean_ms']:8.2f}ms ± {torch_results['batch_std_ms']:.2f}ms")
        print(f"  Per sentence:   {torch_results['per_sentence_ms']:8.3f}ms")
        print(f"  P50/P95/P99:    {torch_results['p50_ms']:.2f} / {torch_results['p95_ms']:.2f} / {torch_results['p99_ms']:.2f}ms")

    print(f"\n[NumPy/ONNX]")
    print(f"  Load time:      {numpy_results['load_time_ms']:8.1f}ms")
    print(f"  Per batch:      {numpy_results['batch_mean_ms']:8.2f}ms ± {numpy_results['batch_std_ms']:.2f}ms")
    print(f"  Per sentence:   {numpy_results['per_sentence_ms']:8.3f}ms")
    print(f"  P50/P95/P99:    {numpy_results['p50_ms']:.2f} / {numpy_results['p95_ms']:.2f} / {numpy_results['p99_ms']:.2f}ms")

    if torch_results:
        speedup = torch_results['per_sentence_ms'] / numpy_results['per_sentence_ms']
        print(f"\n[Speedup]")
        print(f"  NumPy is {speedup:.1f}x faster per sentence")


def main():
    parser = argparse.ArgumentParser(description='Benchmark silero-stress implementations')
    parser.add_argument('--numpy-only', action='store_true', help='Only benchmark NumPy (no torch required)')
    parser.add_argument('--lang', choices=['ru', 'ukr', 'bel', 'all'], default='all', help='Language to benchmark')
    parser.add_argument('--no-homosolver', action='store_true', help='Disable homograph solver for RU')
    args = parser.parse_args()

    languages = ['ru', 'ukr', 'bel'] if args.lang == 'all' else [args.lang]
    use_homosolver = not args.no_homosolver

    print("="*70)
    print("SILERO-STRESS BENCHMARK")
    print("="*70)

    has_torch = False
    if not args.numpy_only:
        try:
            import torch
            torch.set_num_threads(1)
            has_torch = True
            print(f"PyTorch: {torch.__version__} (threads=1)")
        except ImportError:
            print("PyTorch: not available (use --numpy-only)")

    import numpy as np
    print(f"NumPy: {np.__version__}")

    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
    except ImportError:
        print("ONNX Runtime: not available")

    results_summary = []

    for lang in languages:
        torch_results = None
        if has_torch:
            print(f"\nBenchmarking PyTorch ({lang})...")
            torch_results = benchmark_torch(lang, use_homosolver=(lang == 'ru' and use_homosolver))

        print(f"Benchmarking NumPy ({lang})...")
        numpy_results = benchmark_numpy(lang, use_homosolver=(lang == 'ru' and use_homosolver))

        print_results(lang, torch_results, numpy_results)

        if torch_results:
            speedup = torch_results['per_sentence_ms'] / numpy_results['per_sentence_ms']
            results_summary.append({
                'lang': lang,
                'torch_ms': torch_results['per_sentence_ms'],
                'numpy_ms': numpy_results['per_sentence_ms'],
                'speedup': speedup,
            })
        else:
            results_summary.append({
                'lang': lang,
                'numpy_ms': numpy_results['per_sentence_ms'],
            })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if has_torch:
        print(f"\n| Lang | PyTorch | NumPy | Speedup |")
        print(f"|------|---------|-------|---------|")
        for r in results_summary:
            print(f"| {r['lang'].upper()} | {r['torch_ms']:.2f}ms | {r['numpy_ms']:.2f}ms | **{r['speedup']:.1f}x** |")
    else:
        print(f"\n| Lang | NumPy (ms/sentence) |")
        print(f"|------|---------------------|")
        for r in results_summary:
            print(f"| {r['lang'].upper()} | {r['numpy_ms']:.2f}ms |")


if __name__ == '__main__':
    main()
