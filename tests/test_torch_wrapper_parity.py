#!/usr/bin/env python3
"""
Comprehensive test suite for silero-stress NumPy/ONNX implementation.

Tests include:
1. Parity tests: NumPy vs PyTorch (reference) implementations
2. Unit tests: N-gram extraction, tokenization, BERT tokenizer
3. Regression tests: Specific expected outputs
4. Edge cases: Hyphenated words, single vowels, mixed case

TorchAccentorRu is the verified-correct reference (100% parity with original 1.4).
"""

import re
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


# =============================================================================
# Unit Tests: Internal Components (no PyTorch required)
# =============================================================================

class TestNgramExtraction(unittest.TestCase):
    """Test n-gram extraction algorithm."""

    def test_basic_ngram(self):
        """Test n-gram generation for a basic word."""
        from silero_stress._impl import word_ngrams

        ngrams = word_ngrams("кот")
        # Should include boundary markers
        self.assertIn("<", ngrams)
        self.assertIn(">", ngrams)
        self.assertIn("<к", ngrams)
        self.assertIn("кот", ngrams)
        self.assertIn("т>", ngrams)
        self.assertIn("<кот>", ngrams)

    def test_short_word_ngrams(self):
        """Test n-gram generation for single-char word."""
        from silero_stress._impl import word_ngrams

        ngrams = word_ngrams("а")
        self.assertIn("<а>", ngrams)
        self.assertIn("<а", ngrams)
        self.assertIn("а>", ngrams)

    def test_ngram_count(self):
        """Test that n-gram count is reasonable."""
        from silero_stress._impl import word_ngrams

        # For "кот" (len=3), max_n = 3+3 = 6
        # With markers "<кот>", we get ngrams of length 1 to 6
        ngrams = word_ngrams("кот")
        self.assertGreater(len(ngrams), 10)
        self.assertLess(len(ngrams), 30)


class TestTokenizationRegex(unittest.TestCase):
    """Test tokenization regex patterns."""

    def test_basic_split(self):
        """Test basic sentence splitting."""
        pattern = re.compile(r'([\s.,!?;:<>=()/\\]+)')
        result = [x for x in pattern.split("Привет, как дела?") if x]
        self.assertEqual(result, ["Привет", ", ", "как", " ", "дела", "?"])

    def test_punctuation_split(self):
        """Test splitting on various punctuation."""
        pattern = re.compile(r'([\s.,!?;:<>=()/\\]+)')
        result = [x for x in pattern.split("A.B.C") if x]
        self.assertEqual(result, ["A", ".", "B", ".", "C"])

    def test_parentheses_split(self):
        """Test splitting with parentheses."""
        pattern = re.compile(r'([\s.,!?;:<>=()/\\]+)')
        result = [x for x in pattern.split("test (value)") if x]
        self.assertEqual(result, ["test", " (", "value", ")"])


class TestHyphenHandling(unittest.TestCase):
    """Test hyphenated word handling."""

    def test_hyphen_split_algorithm(self):
        """Test the hyphen splitting algorithm."""
        test_cases = [
            ("что-то", ["что-", "то"]),
            ("кому-то", ["кому-", "то"]),
            ("где-нибудь", ["где-", "нибудь"]),
            ("кое-как", ["кое-", "как"]),
        ]

        for word, expected in test_cases:
            with self.subTest(word=word):
                subparts = word.split('-')
                if len(subparts) == 1:
                    tokens = subparts
                else:
                    tokens = [p + '-' for p in subparts[:-1]] + [subparts[-1]]
                self.assertEqual(tokens, expected)

    def test_to_suffix_not_stressed(self):
        """Test that -то suffix doesn't get stress (regression test)."""
        from silero_stress._impl import AccentorRu

        acc = AccentorRu(use_homosolver=False)

        # -то should NOT get stress
        self.assertEqual(acc("что-то", put_yo=False), "чт+о-то")
        self.assertEqual(acc("кому-то", put_yo=False), "ком+у-то")
        self.assertEqual(acc("где-нибудь", put_yo=False), "гд+е-ниб+удь")


class TestHomographPattern(unittest.TestCase):
    """Test homograph detection pattern."""

    def test_cyrillic_matching(self):
        """Test pattern matches Cyrillic words."""
        pattern = re.compile(r'(?=.*[а-яёА-ЯЁ])[а-яёА-ЯЁ+]+')

        self.assertEqual(pattern.findall("замок"), ["замок"])
        self.assertEqual(pattern.findall("ЗАМОК"), ["ЗАМОК"])
        self.assertEqual(pattern.findall("зам+ок"), ["зам+ок"])

    def test_non_cyrillic_not_matched(self):
        """Test pattern doesn't match non-Cyrillic."""
        pattern = re.compile(r'(?=.*[а-яёА-ЯЁ])[а-яёА-ЯЁ+]+')

        self.assertEqual(pattern.findall("test"), [])
        self.assertEqual(pattern.findall("123"), [])

    def test_mixed_content(self):
        """Test pattern with mixed content."""
        pattern = re.compile(r'(?=.*[а-яёА-ЯЁ])[а-яёА-ЯЁ+]+')

        self.assertEqual(pattern.findall("Привет, мир!"), ["Привет", "мир"])
        self.assertEqual(pattern.findall("слово1слово"), ["слово", "слово"])


@unittest.skipIf(not HAS_ONNX, "ONNX Runtime required")
class TestBertTokenizer(unittest.TestCase):
    """Test BERT tokenizer implementation."""

    @classmethod
    def setUpClass(cls):
        from silero_stress._impl import SimpleBertTokenizer, _load_gzip_json
        vocab = _load_gzip_json('ru', 'vocab_ru.json.gz')
        special_tokens = _load_gzip_json('ru', 'special_tokens_ru.json.gz')
        cls.tokenizer = SimpleBertTokenizer(vocab, special_tokens)

    def test_encode_adds_special_tokens(self):
        """Test that encode adds CLS and SEP tokens."""
        ids = self.tokenizer.encode("привет")
        self.assertEqual(ids[0], self.tokenizer.cls_token_id)
        self.assertEqual(ids[-1], self.tokenizer.sep_token_id)

    def test_homo_markers_preserved(self):
        """Test that [HOMO] markers are preserved in tokenization."""
        text = "тест [HOMO] слово [/HOMO] конец"
        ids = self.tokenizer.encode(text)
        self.assertIn(self.tokenizer.homo_start_id, ids)
        self.assertIn(self.tokenizer.homo_end_id, ids)

    def test_encode_various_inputs(self):
        """Test encoding various inputs."""
        test_cases = [
            "привет",
            "Привет, как дела?",
            "слово [HOMO] замок [/HOMO] тест",
        ]
        for text in test_cases:
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                self.assertGreater(len(ids), 2)  # At least CLS + word + SEP


# =============================================================================
# Regression Tests: Expected Outputs (no PyTorch required)
# =============================================================================

class TestExpectedOutputsRussian(unittest.TestCase):
    """Test specific expected outputs for Russian."""

    @classmethod
    def setUpClass(cls):
        from silero_stress._impl import AccentorRu
        cls.acc = AccentorRu(use_homosolver=False)

    def test_basic_words(self):
        """Test basic word stress."""
        cases = [
            ("привет", "прив+ет"),
            ("москва", "москв+а"),
            ("Москва", "Москв+а"),
            ("МОСКВА", "МОСКВ+А"),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                self.assertEqual(self.acc(word, put_yo=False), expected)

    def test_single_vowel(self):
        """Test single vowel words."""
        cases = [
            ("кот", "к+от"),
            ("дом", "д+ом"),
            ("да", "д+а"),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                self.assertEqual(self.acc(word, put_yo=False, stress_single_vowel=True), expected)


class TestExpectedOutputsUkrainian(unittest.TestCase):
    """Test specific expected outputs for Ukrainian."""

    @classmethod
    def setUpClass(cls):
        from silero_stress._impl import AccentorUkrBel
        cls.acc = AccentorUkrBel('ukr')

    def test_basic_words(self):
        """Test basic word stress."""
        self.assertEqual(self.acc("привіт"), "прив+іт")
        self.assertEqual(self.acc("так"), "т+ак")

    def test_sentence(self):
        """Test sentence processing."""
        self.assertEqual(self.acc("Привіт, як справи?"), "Прив+іт, +як справ+и?")

    def test_special_chars(self):
        """Test Ukrainian special characters є, ї, ґ."""
        self.assertEqual(self.acc("їжак"), "їж+ак")  # ї at start
        self.assertEqual(self.acc("єдиний"), "єд+иний")  # є at start
        self.assertEqual(self.acc("київ"), "к+иїв")  # ї in middle


class TestExpectedOutputsBelarusian(unittest.TestCase):
    """Test specific expected outputs for Belarusian."""

    @classmethod
    def setUpClass(cls):
        from silero_stress._impl import AccentorUkrBel
        cls.acc = AccentorUkrBel('bel')

    def test_basic_words(self):
        """Test basic word stress."""
        self.assertEqual(self.acc("беларусь"), "белар+усь")
        self.assertEqual(self.acc("так"), "т+ак")

    def test_sentence(self):
        """Test sentence processing."""
        self.assertEqual(self.acc("Добры дзень!"), "Д+обры дз+ень!")

    def test_yo_vowel(self):
        """Test Belarusian ё handling - ё is a vowel and can be stressed."""
        # ё at start
        self.assertEqual(self.acc("ёсць"), "+ёсць")
        self.assertEqual(self.acc("ёлка"), "+ёлка")
        # ё in middle
        self.assertEqual(self.acc("вёска"), "в+ёска")
        self.assertEqual(self.acc("сёння"), "с+ёння")
        self.assertEqual(self.acc("цёплы"), "ц+ёплы")
        self.assertEqual(self.acc("зялёны"), "зял+ёны")
        # ё at end
        self.assertEqual(self.acc("маё"), "ма+ё")

    def test_yo_in_sentence(self):
        """Test ё in Belarusian sentences."""
        self.assertEqual(self.acc("Сёння добры дзень"), "С+ёння д+обры дз+ень")
        self.assertEqual(self.acc("Зялёная ёлка"), "Зял+ёная +ёлка")


# =============================================================================
# Parity Tests: NumPy vs PyTorch (requires both)
# =============================================================================

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

    def test_hyphenated_words_parity(self):
        """Test hyphenated word handling matches PyTorch."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        words = ["что-то", "кому-то", "где-нибудь", "кое-как"]
        for word in words:
            with self.subTest(word=word):
                torch_result = self.torch_acc(word, put_yo=False)
                numpy_result = self.numpy_acc(word, put_yo=False)
                self.assertEqual(torch_result, numpy_result,
                    f"\nTorch: {torch_result}\nNumPy: {numpy_result}")

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

    def test_ukrainian_special_chars_parity(self):
        """Test Ukrainian special characters є, ї, ґ - parity between backends."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        test_cases = [
            "їжак",       # ї at start
            "єдиний",     # є at start
            "київ",       # ї in middle
            "україна",    # ї in word
            "Слава Україні",  # sentence with ї
        ]

        for test in test_cases:
            with self.subTest(test=test):
                torch_result = self.torch_acc(test)
                numpy_result = self.numpy_acc(test)
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

    def test_belarusian_yo_parity(self):
        """Test Belarusian ё handling - parity between backends."""
        if not (self.torch_loaded and self.numpy_loaded):
            self.skipTest("Implementations not loaded")

        # Words with ё in different positions
        test_cases = [
            "ёсць",       # ё at start
            "ёлка",       # ё at start
            "вёска",      # ё in middle
            "сёння",      # ё in middle
            "цёплы",      # ё in middle
            "зялёны",     # ё in middle
            "маё",        # ё at end
            "Сёння добры дзень",  # sentence with ё
            "Зялёная ёлка",       # multiple ё
            "Цёплая вёска",       # multiple ё
        ]

        for test in test_cases:
            with self.subTest(test=test):
                torch_result = self.torch_acc(test)
                numpy_result = self.numpy_acc(test)
                self.assertEqual(torch_result, numpy_result,
                    f"\nTorch: {torch_result}\nNumPy: {numpy_result}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
