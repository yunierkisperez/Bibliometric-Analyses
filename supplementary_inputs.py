"""Validate the additional Table S2 rows used to construct Figure S1."""
import numpy as np
import pandas as pd

UNION_KEY = 'IMFA_or_CIS'


def imfa_or_cis_from_s2(frame):
    """Read an observed union, never the sum of the two marginal counts."""
    keys = ['id-paper', 'analysis_year']
    union = frame.loc[frame['metric_key'].eq(UNION_KEY)].copy()
    if union.empty:
        raise ValueError('Figure S1 requires the IMFA or EC-CIS rows in the expanded Table-S2.xlsx or Table-S2.csv supplied with the article')
    if union.duplicated(keys).any():
        raise ValueError('Table S2: duplicate IMFA or EC-CIS institution-year rows')
    if not union['denominator_column'].eq('N_articles_attributed').all():
        raise ValueError('Table S2: the IMFA or EC-CIS denominator must be N articles attributed')
    count, denominator, rate = (union[c] for c in ['count', 'denominator', 'value_raw'])
    if count.isna().any() or denominator.isna().any() or (count < 0).any() or (count > denominator).any():
        raise ValueError('Table S2: invalid union count or denominator')
    if (count.mod(1).ne(0) | denominator.mod(1).ne(0)).any():
        raise ValueError('Table S2: publication counts must be integers')
    expected = count.div(denominator.replace(0, np.nan))
    if not np.allclose(rate, expected, rtol=5e-14, atol=1e-12, equal_nan=True):
        raise ValueError('Table S2: union rate must equal count/denominator as a fraction, not a percentage')
    for key in ['IMFA', 'CIS']:
        marginal = frame.loc[frame['metric_key'].eq(key)]
        if marginal.duplicated(keys).any():
            raise ValueError('Table S2: duplicate marginal institution-year rows')
        if set(map(tuple, marginal[keys].values)) != set(map(tuple, union[keys].values)):
            raise ValueError('Table S2: union rows must cover the same institution-years as IMFA and EC-CIS')
        check = union.merge(marginal, on=keys, suffixes=('', '_m'), validate='one_to_one')
        if (check['count'] < check['count_m']).any() or not check['denominator'].eq(check['denominator_m']).all():
            raise ValueError('Table S2: union counts or denominators conflict with a marginal indicator')
    for column in ['deviation_side', 'deviation_strength']:
        union[column] = union[column].fillna('')
    return union.rename(columns={'count': 'N_articles_IMFA_or_CIS',
        'denominator': 'N_articles_attributed', 'value_raw': 'IMFA_or_CIS_rate'})[
        keys + ['N_articles_attributed', 'N_articles_IMFA_or_CIS', 'IMFA_or_CIS_rate',
                'robust_D', 'deviation_side', 'deviation_strength']]
