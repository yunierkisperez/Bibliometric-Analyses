"""Use the bundled, unmodified Source Sans 3 fonts for portable vector text."""
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FONT_DIRECTORY = ROOT / 'assets/fonts/source_sans_3'
FAMILY = 'Source Sans 3'


def verify_font_bundle(directory=None):
    """Verify all bundled font and licence bytes against the pinned manifest."""
    directory = Path(directory) if directory is not None else FONT_DIRECTORY
    manifest = json.loads((directory / 'font_manifest.json').read_text())
    expected = {'regular': 'SourceSans3-Regular.ttf', 'semibold': 'SourceSans3-Semibold.ttf',
                'bold': 'SourceSans3-Bold.ttf'}
    if manifest.get('family') != FAMILY or manifest.get('license') != 'OFL-1.1' or manifest.get('faces') != expected:
        raise ValueError('Unexpected bundled font family, licence or face inventory')
    if set(manifest['files']) != {*expected.values(), 'LICENSE.md'}:
        raise ValueError('Unexpected font-resource file inventory')
    for name, metadata in manifest['files'].items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError(f'Bundled font resource is missing or changed: {name}')
    return manifest


@lru_cache(maxsize=3)
def _font(face):
    from fontTools.ttLib import TTFont
    manifest = verify_font_bundle()
    if face not in manifest['faces']:
        raise ValueError(f'Unknown font face: {face}')
    font = TTFont(FONT_DIRECTORY / manifest['faces'][face])
    return font, font.getGlyphSet(), font.getBestCmap(), font['head'].unitsPerEm


@lru_cache(maxsize=8192)
def _glyph(face, character):
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    _, glyph_set, cmap, units = _font(face)
    name = cmap.get(ord(character))
    if name is None:
        raise KeyError(character)
    glyph = glyph_set[name]
    pen = SVGPathPen(glyph_set)
    glyph.draw(TransformPen(pen, (1 / units, 0, 0, 1 / units, 0, 0)))
    return {'d': pen.getCommands(), 'advance': glyph.width / units}


class _Face(Mapping):
    def __init__(self, face):
        self.face = face

    def __getitem__(self, character):
        return _glyph(self.face, character)

    def __iter__(self):
        return (chr(code) for code in _font(self.face)[2])

    def __len__(self):
        return len(_font(self.face)[2])


def load_glyphs():
    """Read outlines lazily from the open fonts; no external glyph file is used."""
    manifest = verify_font_bundle()
    return {face: _Face(face) for face in manifest['faces']}


def character_positions(text, x, size, face='regular'):
    """Give live SVG and outlined text the same explicit glyph positions."""
    positions = []
    for character in text:
        positions.append(x)
        try:
            x += size * _glyph(face, character)['advance']
        except KeyError:
            raise ValueError(f'{FAMILY} {face} does not contain character {character!r}') from None
    return positions or [x]


def configure_matplotlib():
    """Register bundled faces without requiring a system font installation."""
    import matplotlib
    from matplotlib import font_manager
    manifest = verify_font_bundle()
    for name in manifest['faces'].values():
        font_manager.fontManager.addfont(str(FONT_DIRECTORY / name))
    matplotlib.rcParams['font.family'] = FAMILY
