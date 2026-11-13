# Fonts and figure assets

The figures use Source Sans 3, version 3.052, by Adobe / Paul D. Hunt. Regular,
Semibold and Bold are bundled as unmodified TrueType files under SIL Open Font
License 1.1. The upstream copyright and Reserved Font Name notice are retained
in `assets/fonts/source_sans_3/LICENSE.md`.

The font manifest records upstream tag `3.052R`, commit
`ed1808970eb3c7301c9a523bee26473ba0bb62fa`, and the SHA-256 hashes of the fonts
and licence. `font_resources.py` reads outlines and advances directly from these
files. Run `python tools/check_installation.py` to verify the bundle.

Keep the complete font directory and its licence when redistributing the
software. The fonts retain OFL-1.1 independently of the Python code licence.
The SVG panel designs contain axes, labels and placement information; data
marks are drawn from the input tables during rendering.

PDF and master SVG exports contain vector outlines. The `_text.svg` companion
retains character-editable text. Install the three supplied fonts when editing
that companion to preserve its typography.

Sources:

- https://github.com/adobe-fonts/source-sans/releases/tag/3.052R
- https://github.com/adobe-fonts/source-sans/blob/ed1808970eb3c7301c9a523bee26473ba0bb62fa/LICENSE.md
