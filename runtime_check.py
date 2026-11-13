"""Check local runtime dependencies before starting an expensive calculation."""
import importlib
import importlib.metadata
import platform

PACKAGES = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy',
    'matplotlib': 'matplotlib', 'PIL': 'Pillow', 'openpyxl': 'openpyxl',
    'fontTools': 'fonttools', 'fitz': 'PyMuPDF', 'cairosvg': 'CairoSVG'}


def check(figures=True):
    versions = {}
    names = PACKAGES if figures else {k: v for k, v in PACKAGES.items() if k not in ['fitz', 'cairosvg']}
    for module, distribution in names.items():
        try:
            importlib.import_module(module)
            versions[distribution] = importlib.metadata.version(distribution)
        except (ImportError, OSError) as error:
            raise RuntimeError(f'Cannot load {distribution}: {error}. See INSTALL.md; activate the documented environment before running this command.') from error
    if figures:
        import cairosvg
        # Exercise the installed native Cairo, without writing an output file.
        pdf = cairosvg.svg2pdf(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><path d="M0 0L1 1"/></svg>')
        if not pdf.startswith(b'%PDF-'):
            raise RuntimeError('CairoSVG did not produce a PDF; see INSTALL.md')
        from font_resources import verify_font_bundle
        verify_font_bundle()
    return {'python': platform.python_version(), 'packages': versions,
        'figure_dependencies_checked': figures, 'native_cairo_exercised': figures}
