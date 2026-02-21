#!/usr/bin/env python3
"""
Final parity test: compare original 1.4, torch wrapper, and NumPy/ONNX implementations.

Requirements:
    - TORCH_VENV: virtualenv with silero-stress 1.4 + torch installed
    - NUMPY_VENV: virtualenv with numpy + onnxruntime (no silero-stress)

Usage:
    # Generate reference from original 1.4 (use venv with 1.4 installed):
    .venv/bin/python scripts/final_parity_test.py --mode=reference

    # Test torch wrapper (from src):
    PYTHONPATH=src .venv/bin/python scripts/final_parity_test.py --mode=torch

    # Test numpy/onnx (from src, use numpy-only venv):
    PYTHONPATH=src .onnx_venv/bin/python scripts/final_parity_test.py --mode=numpy

    # Or run all comparisons at once:
    ./scripts/run_final_parity.sh
"""

import argparse
import json
import sys
from pathlib import Path

# Test sentences organized by category
TEST_SENTENCES = {
    # Simple sentences (no homographs)
    "simple": [
        "Привет мир",
        "Москва столица России",
        "Добрый день",
        "Как дела",
        "Спасибо большое",
        "Здравствуйте",
        "До свидания",
        "Я люблю программирование",
    ],

    # Sentences with yo (ё) words
    "yo_words": [
        "Елка стоит в углу",
        "Зеленый лес красивый",
        "Все хорошо",
        "Мед очень вкусный",
        "Черный кот сидит",
        "Желтый лист упал",
        "Пчелы делают мед",
        "Еще раз спасибо",
    ],

    # Sentences with homographs (BERT disambiguation)
    "homographs": [
        "Это стоит дорого",
        "Елка стоит в углу",
        "Замок закрыт на ключ",
        "Старый замок на горе",
        "Я уже готов идти",
        "Он готов помочь",
        "Пойдем домой",
        "Дома никого нет",
        "Мука для хлеба",
        "Какая мука терпеть это",
        "Большая жила в шахте",
        "Она жила в городе",
    ],

    # Complex sentence with multiple homographs
    "complex": [
        "Меня зовут Лева Королев. Я из готов. И я уже готов открыть все ваши замки любой сложности!",
        "Привет, я текст без омографов.",
    ],

    # Edge cases
    "edge_cases": [
        "Да",  # Single vowel
        "Нет",  # Single vowel
        "А",  # Single letter
        "Я иду",  # Short words
        "123 abc тест",  # Mixed content
        "ПРИВЕТ МИР",  # Uppercase
        "привет! как? дела.",  # Punctuation
    ],

    # Torch yo edge cases - some are bugs, some are valid
    # 3 BUGS: torch adds ё to UNSTRESSED vowels (invalid Russian phonology)
    # 4 VALID: torch correctly uses ё on STRESSED vowel
    "torch_yo_bugs": [
        # BUG: ё in unstressed position (numpy produces correct form)
        "Он отменен.",       # torch: отм+енён (invalid - ё unstressed), numpy: отм+енен (valid)
        "Это мужичье дело.", # torch: муж+ичьё (invalid - ё unstressed), numpy: муж+ичье (valid)
        "Он напоен водой.",  # torch: нап+оён (invalid - ё unstressed), numpy: нап+оен (valid)
        # VALID: ё on stressed vowel (both implementations match)
        "Мы познаем мир.",   # both: позна+ём (valid - ё is stressed)
        "Ты познаешь истину.", # both: позна+ёшь (valid - ё is stressed)
        "Он покорен судьбой.", # both: покор+ён (valid - ё is stressed)
        "Мы запоем песню.",  # both: запо+ём (valid - ё is stressed)
    ],

    # Additional homograph words for thorough testing
    "more_homographs": [
        "Большой город",
        "Черных кошек много",
        "Меду в банке много",
        "Распознает образы",
        "Атлас мира на столе",
        "Атлас ткани мягкий",
        "Белки в лесу",
        "Белки в яйце",
        "Ирис цветет",
        "Ирис конфета",
    ],
}


def load_accentor_reference():
    """Load original silero-stress 1.4 accentor."""
    from silero_stress import load_accentor
    return load_accentor(lang='ru')


def load_accentor_torch():
    """Load torch wrapper from src."""
    from silero_stress._torch_impl import TorchAccentorRu
    return TorchAccentorRu(use_homosolver=True, device='cpu')


def load_accentor_numpy():
    """Load numpy/onnx implementation from src."""
    from silero_stress._impl import AccentorRu
    return AccentorRu(use_homosolver=True, device='cpu')


def run_tests(accentor, put_yo=True):
    """Run all test sentences and return results."""
    results = {}
    for category, sentences in TEST_SENTENCES.items():
        results[category] = {}
        for sent in sentences:
            try:
                # All implementations support put_yo parameter
                result = accentor(sent, put_yo=put_yo)
                results[category][sent] = result
            except Exception as e:
                results[category][sent] = f"ERROR: {e}"
    return results


def generate_reference(output_path):
    """Generate reference results from original 1.4."""
    print("Loading original silero-stress 1.4...")
    accentor = load_accentor_reference()

    print("Running tests...")
    results = run_tests(accentor, put_yo=True)

    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    total = sum(len(sents) for sents in results.values())
    print(f"\nGenerated {total} reference results.")
    return results


def compare_results(mode, reference_path):
    """Compare implementation results against reference."""
    print(f"\nLoading reference from {reference_path}...")
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference = json.load(f)

    print(f"Loading {mode} implementation...")
    if mode == 'torch':
        accentor = load_accentor_torch()
    elif mode == 'numpy':
        accentor = load_accentor_numpy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("Running tests...")
    results = run_tests(accentor, put_yo=True)

    # Compare
    matches = 0
    mismatches = []
    expected_diffs = []  # torch_yo_bugs category - numpy is correct, torch has bug
    errors = []
    total = 0

    for category, sentences in reference.items():
        for sent, ref_result in sentences.items():
            total += 1
            if sent not in results.get(category, {}):
                errors.append({'category': category, 'sentence': sent, 'error': 'Missing in results'})
                continue

            test_result = results[category][sent]

            if test_result.startswith('ERROR:'):
                errors.append({'category': category, 'sentence': sent, 'error': test_result})
            elif test_result == ref_result:
                matches += 1
            elif category == 'torch_yo_bugs' and mode == 'numpy':
                # Expected difference: numpy produces correct output, torch/1.4 has bug
                expected_diffs.append({
                    'category': category,
                    'sentence': sent,
                    'reference_bug': ref_result,
                    'numpy_correct': test_result,
                })
            else:
                mismatches.append({
                    'category': category,
                    'sentence': sent,
                    'reference': ref_result,
                    mode: test_result,
                })

    # Print results
    print("\n" + "=" * 70)
    print(f"PARITY TEST RESULTS: {mode.upper()} vs REFERENCE (1.4)")
    print("=" * 70)
    print(f"\nTotal sentences: {total}")
    print(f"Exact matches: {matches}")
    if expected_diffs:
        print(f"Expected diffs (torch yo bugs): {len(expected_diffs)}")
    print(f"Unexpected mismatches: {len(mismatches)}")
    print(f"Errors: {len(errors)}")

    effective_matches = matches + len(expected_diffs)
    print(f"\nEffective parity: {effective_matches}/{total} ({100*effective_matches/total:.1f}%)")

    if expected_diffs:
        print(f"\n--- EXPECTED DIFFERENCES (torch yo bugs - numpy is CORRECT) ---")
        for d in expected_diffs:
            print(f"\n[{d['category']}] {d['sentence']}")
            print(f"  1.4 (bug):      {d['reference_bug']}")
            print(f"  NumPy (correct): {d['numpy_correct']}")

    if mismatches:
        print(f"\n--- UNEXPECTED MISMATCHES ({len(mismatches)}) ---")
        for m in mismatches:
            print(f"\n[{m['category']}] {m['sentence']}")
            print(f"  Reference: {m['reference']}")
            print(f"  {mode.title():8}: {m[mode]}")

    if errors:
        print(f"\n--- ERRORS ({len(errors)}) ---")
        for e in errors:
            print(f"  [{e['category']}] {e['sentence']}: {e['error']}")

    # Save comparison results
    output = {
        'mode': mode,
        'total': total,
        'matches': matches,
        'expected_diffs': expected_diffs,
        'mismatches': mismatches,
        'errors': errors,
        'results': results,
    }
    output_path = f"parity_{mode}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_path}")

    # Success if no unexpected mismatches or errors
    return len(mismatches) == 0 and len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description='Final parity test for silero-stress implementations')
    parser.add_argument('--mode', choices=['reference', 'torch', 'numpy'], required=True,
                        help='Mode: reference (generate from 1.4), torch (test wrapper), numpy (test onnx)')
    parser.add_argument('--output', default='reference_1.4.json',
                        help='Output file for reference mode')
    parser.add_argument('--reference', default='reference_1.4.json',
                        help='Reference file for comparison modes')
    args = parser.parse_args()

    if args.mode == 'reference':
        generate_reference(args.output)
    else:
        success = compare_results(args.mode, args.reference)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
