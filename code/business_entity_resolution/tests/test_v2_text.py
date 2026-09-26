import unittest

from src.v2_text import clean_address, clean_name, entity_partition, grams, normalize


class UnicodeRetrievalTests(unittest.TestCase):
    def test_keeps_devanagari_letters_and_vowel_marks(self):
        self.assertEqual(normalize("राम मार्केटिंग"), "राम मार्केटिंग")
        self.assertTrue(grams(normalize("राम मार्केटिंग")))

    def test_folds_accents_without_losing_french_name(self):
        self.assertEqual(clean_name("Société Étoile SARL"), "societe etoile")

    def test_removes_website_noise_and_normalizes_address(self):
        self.assertEqual(clean_name("ACME Pvt Ltd | www.acme.com"), "acme")
        self.assertEqual(clean_address("12 North Road, Suite 4"), "12 n rd ste 4")

    def test_grams_cover_interior_spelling_evidence(self):
        self.assertIn("ket", grams("marketing"))
        self.assertGreater(len(grams("a reasonably long business name")), 4)

    def test_fold_is_deterministic(self):
        self.assertEqual(entity_partition("S1-123", 500), entity_partition("S1-123", 500))


if __name__ == "__main__":
    unittest.main()
