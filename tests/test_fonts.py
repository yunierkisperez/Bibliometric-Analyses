"""Portable open-font controls using only fictitious labels and text."""
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from font_resources import FONT_DIRECTORY, load_glyphs, verify_font_bundle
from vector_composition import SVG, add_text, outline_text


class OpenFontControls(unittest.TestCase):
    def test_outlined_unicode_text_needs_only_the_bundled_open_fonts(self):
        root = ET.Element('{' + SVG + '}svg')
        add_text(root, 'Fictitious U-01: Δ ≤ −2; D ≥ 3', 10, 20, 10)
        outline_text(root, load_glyphs())
        self.assertFalse(root.findall('.//{' + SVG + '}text'))
        self.assertTrue(root.findall('.//{' + SVG + '}path'))
        self.assertFalse(root.findall('.//{' + SVG + '}image'))
        self.assertEqual(next(iter(root)).get('data-label'), 'Fictitious U-01: Δ ≤ −2; D ≥ 3')

    def test_changed_font_bytes_fail_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'fonts'
            shutil.copytree(FONT_DIRECTORY, source)
            with (source / 'SourceSans3-Regular.ttf').open('ab') as stream:
                stream.write(b'fictitious corruption')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                verify_font_bundle(source)

    def test_all_three_unmodified_faces_and_their_licence_are_bundled(self):
        manifest = verify_font_bundle()
        self.assertEqual(manifest['license'], 'OFL-1.1')
        self.assertEqual(set(manifest['faces']), {'regular', 'semibold', 'bold'})
        self.assertIn('SIL OPEN FONT LICENSE', (FONT_DIRECTORY / 'LICENSE.md').read_text())


if __name__ == '__main__':
    unittest.main()
