"""Public product identifiers and the dependencies of independent workflow steps."""
import re

YEARS = tuple(range(2011, 2026))
ANALYSES = {
    'growth': (1, 'plot_fig1_national_growth_robust_d.py', ('institution_year_metrics',)),
    'exposure': (2, 'plot_fig2_temporal_exposure_profiles.py',
                 ('institution_year_metrics', 'hyperprolific_author_flags', 'author_productivity')),
    'composition': (3, 'plot_fig3_compositional_change.py',
                    ('institution_year_metrics', 'national_baseline', 'article_institution_flags',
                     'hyperprolific_author_flags', 'author_productivity')),
    'concentration': (4, 'plot_fig4_author_concentration.py',
                      ('institution_year_metrics', 'top_author_concentration', 'author_productivity', 'hyperprolific_author_flags')),
    'sharing': (5, 'plot_fig5_national_sharing.py',
                ('article_affiliations', 'article_institution_flags', 'author_affiliations',
                 'corresponding_author_affiliations', 'institutions')),
    'affiliations': (6, 'plot_fig6_affiliation_structures.py',
                     ('institution_year_metrics', 'imfa_author_partner_institutions', 'cis_author_partner_institutions')),
    'signals': (7, 'plot_fig7_exposure_signal_matrix.py', ('institution_year_metrics',)),
    'temporal': (8, 'plot_supplementary_temporal_deviations.py', ('institution_year_metrics',)),
}
TABLE_ANALYSIS = {
    **{f'S{n}': group for group, numbers in [
        ('growth', [1]), ('temporal', [2, 3]), ('composition', [4, 5, 6]),
        ('concentration', [7, 8, 9]), ('sharing', [10, 11, 12]), ('affiliations', [13, 14]),
        ('signals', [15])] for n in numbers},
    'Main1': 'composition',
}
# Direct inputs are distinguished from the other manuscript tables associated
# with a figure. Shared tables always retain their original S-number.
FIGURES = {
    'Fig1': (1, ('S1',), ('S1',)),
    'Fig2': (2, ('S2',), ('S2',)),
    'Fig3': (3, ('S5',), ('S4', 'S5', 'S6', 'Main1')),
    'Fig4': (4, ('S7',), ('S7', 'S9')),
    'Fig5': (5, ('S10', 'S11'), ('S10', 'S11')),
    'Fig6': (6, ('S13',), ('S13', 'S14')),
    'Fig7': (7, ('S15',), ('S15',)),
    'Fig-S1': (8, ('S2',), ('S2',)),
    'Fig-S2': (8, ('S3',), ('S3',)),
    'Fig-S3': (3, ('S5',), ('S5',)),
    'Fig-S4': (3, ('S6',), ('S6',)),
    'Fig-S5': (3, ('S6',), ('S6',)),
    'Fig-S6': (4, ('S8',), ('S8',)),
    'Fig-S7': (5, ('S11',), ('S11',)),
    'Fig-S8': (5, ('S12',), ('S12',)),
    'Fig-S9': (6, ('S13', 'S14'), ('S13', 'S14')),
    'Fig-S10': (6, ('S13',), ('S13',)),
}

def figure_products(figure, only_tables=None, minimal_tables=False):
    figure=figure_selection([figure])[0]
    tables=FIGURES[figure][2]
    if only_tables is not None:
        requested = table_selection(only_tables)
        if set(requested).difference(tables):
            raise ValueError(f'{figure} supports table products {", ".join(tables)}')
        return [], requested
    return [figure], list(FIGURES[figure][1] if minimal_tables else tables)


def table_selection(values=None, extras=True):
    allowed = list(TABLE_ANALYSIS) if extras else [f'S{n}' for n in range(1, 16)]
    if values is None or values == ['all']:
        return allowed
    result = []
    for raw in values:
        text = str(raw).strip()
        if re.fullmatch(r'(?:Table-?)?S?(\d+)', text, re.I):
            key = 'S' + str(int(re.search(r'\d+', text)[0]))
        elif text.lower() in ['main1', 'main-table1']:
            key = 'Main1'
        else:
            key = text
        if key not in allowed:
            raise ValueError(f'Unknown table {raw!r}; choose from {", ".join(allowed)}')
        if key in result:
            raise ValueError(f'Duplicate table selection: {key}')
        result.append(key)
    if not result:
        raise ValueError('Select at least one table')
    return result


def figure_selection(values=None):
    if values is None or values == ['all']:
        return list(FIGURES)
    result = []
    for raw in values:
        text = str(raw).strip().lower().replace('figure', '').replace('fig', '').replace('-', '')
        key = 'Fig-S' + str(int(text[1:])) if re.fullmatch(r's\d+', text) else 'Fig' + str(int(text)) if text.isdigit() else str(raw)
        if key not in FIGURES:
            raise ValueError(f'Unknown figure {raw!r}; choose from {", ".join(FIGURES)}')
        if key in result:
            raise ValueError(f'Duplicate figure selection: {key}')
        result.append(key)
    if not result:
        raise ValueError('Select at least one figure')
    return result


def analysis_selection(analyses=None, tables=None, figures=None):
    if tables is not None:
        selected = {TABLE_ANALYSIS[t] for t in table_selection(tables)}
    elif figures is not None:
        selected = {TABLE_ANALYSIS[t] for f in figure_selection(figures) for t in FIGURES[f][2]}
    else:
        selected = set(ANALYSES if analyses is None or analyses == ['all'] else analyses)
    unknown = selected.difference(ANALYSES)
    if unknown or not selected:
        raise ValueError(f'Unknown or empty analysis selection: {sorted(unknown)}')
    if 'temporal' in selected:
        selected.add('exposure')
    return [name for name in ANALYSES if name in selected]
