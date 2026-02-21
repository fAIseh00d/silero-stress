#!/usr/bin/env python3
"""
Parity tests: NumPy/ONNX implementation vs TorchAccentorRu wrapper.

TorchAccentorRu is the verified-correct reference (100% parity with original 1.4).
This test ensures the NumPy/ONNX implementation produces identical results.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Check dependencies
try:
    import torch
    torch.set_num_threads(1)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@unittest.skipIf(not HAS_TORCH, "PyTorch required")
@unittest.skipIf(not HAS_ONNX, "ONNX Runtime required")
class TestRussianAccentorParity(unittest.TestCase):
    """Test AccentorRu (NumPy/ONNX) against TorchAccentorRu."""

    @classmethod
    def setUpClass(cls):
        # Load torch wrapper (reference implementation)
        try:
            from silero_stress._torch_impl import TorchAccentorRu
            cls.torch_acc = TorchAccentorRu(use_homosolver=True, device='cpu')
            cls.torch_loaded = True
        except Exception as e:
            cls.torch_loaded = False
            cls.torch_error = str(e)

        # Load numpy/onnx implementation
        try:
            from silero_stress._impl import AccentorRu
            cls.numpy_acc = AccentorRu(use_homosolver=True, device='cpu')
            cls.numpy_loaded = True
        except Exception as e:
            cls.numpy_loaded = False
            cls.numpy_error = str(e)

    def test_both_loaded(self):
        """Ensure both implementations loaded."""
        if not self.torch_loaded:
            self.fail(f"TorchAccentorRu failed to load: {self.torch_error}")
        if not self.numpy_loaded:
            self.fail(f"AccentorRu failed to load: {self.numpy_error}")

    def test_simple_sentences(self):
        """Test simple sentences without homographs."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        sentences = [
            "Привет мир",
            "Москва столица России",
            "Добрый день",
            "Как дела",
            "Спасибо большое",
        ]

        for sent in sentences:
            with self.subTest(sentence=sent):
                torch_result = self.torch_acc(sent, put_yo=False)
                numpy_result = self.numpy_acc(sent, put_yo=False)
                self.assertEqual(torch_result, numpy_result,
                    f"\nTorch: {torch_result}\nNumPy: {numpy_result}")

    def test_sentences_with_yo(self):
        """Test yo replacement."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        sentences = [
            "Елка стоит",
            "Зеленый лес",
            "Все хорошо",
            "Мед очень вкусный",
        ]

        for sent in sentences:
            with self.subTest(sentence=sent):
                torch_result = self.torch_acc(sent, put_yo=True)
                numpy_result = self.numpy_acc(sent, put_yo=True)
                self.assertEqual(torch_result, numpy_result,
                    f"\nTorch: {torch_result}\nNumPy: {numpy_result}")

    def test_homograph_sentences(self):
        """Test sentences with homographs (BERT disambiguation)."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        sentences = [
            "Елка стоит в углу",
            "Это стоит дорого",
            "Замок закрыт на ключ",
            "Я уже готов идти",
            "Он готов помочь",
            "Пойдем домой",
            "Дома никого нет",
        ]

        for sent in sentences:
            with self.subTest(sentence=sent):
                torch_result = self.torch_acc(sent, put_yo=True)
                numpy_result = self.numpy_acc(sent, put_yo=True)
                self.assertEqual(torch_result, numpy_result,
                    f"\nTorch: {torch_result}\nNumPy: {numpy_result}")

    def test_complex_sentence(self):
        """Test the complex sentence with multiple homographs."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        sent = "Меня зовут Лева Королев. Я из готов. И я уже готов открыть все ваши замки любой сложности!"
        torch_result = self.torch_acc(sent, put_yo=True)
        numpy_result = self.numpy_acc(sent, put_yo=True)
        self.assertEqual(torch_result, numpy_result,
            f"\nTorch: {torch_result}\nNumPy: {numpy_result}")

    def test_stress_single_vowel(self):
        """Test stress_single_vowel parameter."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        # With stress on single vowel
        torch_result = self.torch_acc("да", stress_single_vowel=True, put_yo=False)
        numpy_result = self.numpy_acc("да", stress_single_vowel=True, put_yo=False)
        self.assertEqual(torch_result, numpy_result)

        # Without stress on single vowel
        torch_result = self.torch_acc("да", stress_single_vowel=False, put_yo=False)
        numpy_result = self.numpy_acc("да", stress_single_vowel=False, put_yo=False)
        self.assertEqual(torch_result, numpy_result)

    def test_all_homographs_minimal_context(self):
        """Test all homographs with minimal context.

        Known differences:
        - 7 torch bugs where it adds ё to unstressed vowels (violates Russian ё rules)
          NumPy produces CORRECT output (valid homodict variant) for these cases.
        """
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        # Known torch bugs: torch adds ё to unstressed vowels, creating invalid forms
        # NumPy produces correct output (valid homodict variant) for all of these
        # The ё letter MUST be stressed in Russian, so torch is wrong here.
        known_torch_yo_bugs = {
            'отменен',    # torch: отм+енён (invalid), numpy: отм+енен (valid)
            'познаем',    # torch: позн+аём (invalid), numpy: позн+аем (valid)
            'познаешь',   # torch: позн+аёшь (invalid), numpy: позн+аешь (valid)
            'покорен',    # torch: пок+орён (invalid), numpy: пок+орен (valid)
            'запоем',     # torch: зап+оём (invalid), numpy: зап+оем (valid)
            'мужичье',    # torch: муж+ичьё (invalid), numpy: муж+ичье (valid)
            'напоен',     # torch: нап+оён (invalid), numpy: нап+оен (valid)
        }

        # Get homodict from torch model
        homodict = self.torch_acc._model.homosolver.homodict

        matches = 0
        mismatches = []

        for word in homodict.keys():
            sent = f"Он {word}."

            torch_result = self.torch_acc(sent, put_yo=True)
            numpy_result = self.numpy_acc(sent, put_yo=True)

            if torch_result == numpy_result:
                matches += 1
            elif word in known_torch_yo_bugs:
                # Known torch bug - numpy produces correct output
                # Verify stress positions still match (only yo differs due to torch bug)
                torch_no_yo = self.torch_acc(sent, put_yo=False)
                numpy_no_yo = self.numpy_acc(sent, put_yo=False)
                if torch_no_yo == numpy_no_yo:
                    matches += 1  # Count as match - torch bug, numpy correct
                else:
                    mismatches.append({'word': word, 'torch': torch_result, 'numpy': numpy_result})
            else:
                mismatches.append({
                    'word': word,
                    'torch': torch_result,
                    'numpy': numpy_result
                })

        total = len(homodict)
        print(f"\n\nHomograph parity: {matches}/{total} ({100*matches/total:.1f}%)")

        if mismatches:
            print(f"Mismatches ({len(mismatches)}):")
            for m in mismatches[:20]:
                print(f"  {m['word']}: Torch='{m['torch']}' vs NumPy='{m['numpy']}'")

        self.assertEqual(matches, total,
            f"Match rate: {matches}/{total}. First mismatches: {mismatches[:5]}")


@unittest.skipIf(not HAS_TORCH, "PyTorch required")
class TestUkrainianAccentorParity(unittest.TestCase):
    """Test AccentorUkrBel (NumPy/ONNX) against TorchAccentorUkrBel for Ukrainian."""

    @classmethod
    def setUpClass(cls):
        try:
            from silero_stress._torch_impl import TorchAccentorUkrBel
            cls.torch_acc = TorchAccentorUkrBel('ukr')
            cls.torch_loaded = True
        except Exception as e:
            cls.torch_loaded = False
            cls.torch_error = str(e)

        try:
            from silero_stress._impl import AccentorUkrBel
            cls.numpy_acc = AccentorUkrBel('ukr')
            cls.numpy_loaded = True
        except Exception as e:
            cls.numpy_loaded = False
            cls.numpy_error = str(e)

    def test_both_loaded(self):
        if not self.torch_loaded:
            self.fail(f"TorchAccentorUkrBel failed to load: {self.torch_error}")
        if not self.numpy_loaded:
            self.fail(f"AccentorUkrBel failed to load: {self.numpy_error}")

    def test_ukrainian_sentences(self):
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        sentences = [
            "Привіт світ",
            "Київ столиця України",
            "Добрий день",
            "Дякую дуже",
            "Як справи",
        ]

        for sent in sentences:
            with self.subTest(sentence=sent):
                torch_result = self.torch_acc(sent)
                numpy_result = self.numpy_acc(sent)
                self.assertEqual(torch_result, numpy_result,
                    f"\nTorch: {torch_result}\nNumPy: {numpy_result}")


@unittest.skipIf(not HAS_TORCH, "PyTorch required")
class TestBelarusianAccentorParity(unittest.TestCase):
    """Test AccentorUkrBel (NumPy/ONNX) against TorchAccentorUkrBel for Belarusian."""

    @classmethod
    def setUpClass(cls):
        try:
            from silero_stress._torch_impl import TorchAccentorUkrBel
            cls.torch_acc = TorchAccentorUkrBel('bel')
            cls.torch_loaded = True
        except Exception as e:
            cls.torch_loaded = False
            cls.torch_error = str(e)

        try:
            from silero_stress._impl import AccentorUkrBel
            cls.numpy_acc = AccentorUkrBel('bel')
            cls.numpy_loaded = True
        except Exception as e:
            cls.numpy_loaded = False
            cls.numpy_error = str(e)

    def test_both_loaded(self):
        if not self.torch_loaded:
            self.fail(f"TorchAccentorUkrBel failed to load: {self.torch_error}")
        if not self.numpy_loaded:
            self.fail(f"AccentorUkrBel failed to load: {self.numpy_error}")

    def test_belarusian_sentences(self):
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        sentences = [
            "Прывітанне свет",
            "Мінск сталіца Беларусі",
            "Добры дзень",
            "Дзякуй вельмі",
            "Як справы",
        ]

        for sent in sentences:
            with self.subTest(sentence=sent):
                torch_result = self.torch_acc(sent)
                numpy_result = self.numpy_acc(sent)
                self.assertEqual(torch_result, numpy_result,
                    f"\nTorch: {torch_result}\nNumPy: {numpy_result}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
