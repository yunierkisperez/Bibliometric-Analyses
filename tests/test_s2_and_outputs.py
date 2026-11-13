"""Synthetic controls for the expanded Table S2 and minimal private exports."""
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from annual_outputs import CONTRACT, project
from product_catalog import ANALYSES, analysis_selection, figure_products
from supplementary_inputs import imfa_or_cis_from_s2
from table_export import build, read_table, ADAPTERS, SCHEMAS
from test_pipeline import record, author, A, INSTITUTIONS
import scopus_process as processor


def calculated_sources(folder):
    rows = []
    for metric in ['IMFA', 'CIS', 'HPA', 'HAO', 'NLO', 'HPA_MFA_or_CIS_articles']:
        for code in ['U-01', 'National']:
            rows.append({'id-paper': code, 'analysis_year': 2025, 'metric_key': metric,
                'denominator_column': 'CA_known' if metric == 'NLO' else 'N_articles_attributed',
                'count': 20 if metric == 'IMFA' else 25, 'denominator': 100,
                'value_raw': .2 if metric == 'IMFA' else .25,
                'robust_D': '', 'deviation_side': '', 'deviation_strength': ''})
    pd.DataFrame(rows).to_csv(folder / ADAPTERS['2']['sources'][0], index=False)
    pd.DataFrame([{'id-paper': code, 'analysis_year': 2025,
        'N_articles_attributed': 100, 'N_articles_IMFA_or_CIS': 30,
        'IMFA_or_CIS_rate': .3, 'robust_D': '', 'deviation_side': '',
        'deviation_strength': ''} for code in ['U-01', 'National']]).to_csv(folder / 'Source-Fig-S1.csv', index=False)


class ExpandedTableControls(unittest.TestCase):
    def table(self, root):
        source = root / 'calculated'; source.mkdir()
        calculated_sources(source)
        dest = root / 'tables'
        build(source, dest, tables=[2])
        return dest, read_table(dest, 2)

    def test_s1_requires_s2_and_the_union_analysis(self):
        self.assertEqual(figure_products('Fig-S1'), (['Fig-S1'], ['S2']))
        self.assertEqual(analysis_selection(tables=['S2']), ['exposure', 'temporal'])

    def test_excel_and_csv_round_trip_keep_schema_and_observed_union(self):
        with tempfile.TemporaryDirectory() as temp:
            dest, xlsx = self.table(Path(temp))
            csv = read_table(dest, 2, 'csv')
            self.assertEqual({p.name for p in dest.iterdir()}, {'Table-S2.xlsx', 'Table-S2.csv'})
            self.assertEqual(len(xlsx), 14)
            self.assertEqual(len(xlsx.columns), len(SCHEMAS['2']['headers']))
            left, right = imfa_or_cis_from_s2(xlsx), imfa_or_cis_from_s2(csv)
            pd.testing.assert_frame_equal(left, right, check_dtype=False)
            self.assertEqual(left['N_articles_IMFA_or_CIS'].tolist(), [30, 30])
            self.assertTrue(left['robust_D'].isna().all())
            original = xlsx.loc[xlsx['metric_key'].eq('IMFA')]
            self.assertEqual(original['count'].tolist(), [20, 20])

    def test_missing_union_cannot_be_inferred_from_marginals(self):
        with tempfile.TemporaryDirectory() as temp:
            _, frame = self.table(Path(temp))
            with self.assertRaisesRegex(ValueError, 'requires the IMFA or EC-CIS rows'):
                imfa_or_cis_from_s2(frame.loc[frame['metric_key'].ne('IMFA_or_CIS')])

    def test_duplicate_union_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, frame = self.table(Path(temp))
            duplicate = pd.concat([frame, frame.loc[frame['metric_key'].eq('IMFA_or_CIS')]])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                imfa_or_cis_from_s2(duplicate)

    def test_partial_union_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, frame = self.table(Path(temp))
            frame = frame.loc[~(frame['metric_key'].eq('IMFA_or_CIS') & frame['id-paper'].eq('National'))]
            with self.assertRaisesRegex(ValueError, 'same institution-years'):
                imfa_or_cis_from_s2(frame)

    def test_percent_in_fraction_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, frame = self.table(Path(temp))
            frame.loc[frame['metric_key'].eq('IMFA_or_CIS'), 'value_raw'] = 30
            with self.assertRaisesRegex(ValueError, 'not a percentage'):
                imfa_or_cis_from_s2(frame)


class OutputContractControls(unittest.TestCase):
    def test_unused_top_k_and_national_rows_are_omitted(self):
        tables = {name: pd.DataFrame(columns=spec['columns']) for name, spec in CONTRACT.items()}
        tables['top_author_concentration'] = pd.DataFrame([
            {**{col: 1 for col in CONTRACT['top_author_concentration']['columns']},
             'institution_id': code, 'n_top_authors': k}
            for code in ['FICTITIOUS', 'NATIONAL'] for k in range(1, 11)])
        result = project(tables)['top_author_concentration']
        self.assertEqual(result['n_top_authors'].tolist(), [1, 5, 10])
        self.assertEqual(set(result['institution_id']), {'FICTITIOUS'})

    def test_removing_audit_output_does_not_change_retained_calculations(self):
        old = processor.INSTITUTIONS
        processor.INSTITUTIONS = INSTITUTIONS
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'synthetic_2025.json'
                path.write_text(json.dumps([record(f'SYN-AUDIT-{i}',
                    [author(1, [A]), author(2, [A, '99000003'])]) for i in range(25)]))
                with_audit = project(processor.process_input(path, analysis_year=2025, include_audit=True))
                without_audit = project(processor.process_input(path, analysis_year=2025, include_audit=False))
                for name in CONTRACT:
                    pd.testing.assert_frame_equal(with_audit[name], without_audit[name])
        finally:
            processor.INSTITUTIONS = old

    def test_every_retained_family_has_a_downstream_consumer(self):
        self.assertEqual(set(CONTRACT), set(processor.RETAINED_OUTPUTS))
        for name, spec in CONTRACT.items():
            self.assertTrue(spec['purpose'])
            self.assertTrue(spec['consumers'])
            for consumer in spec['consumers']:
                self.assertIn(name, ANALYSES[consumer][2])
        self.assertNotIn('title', CONTRACT['article_institution_flags']['columns'])
        self.assertNotIn('doi', CONTRACT['article_institution_flags']['columns'])

    def test_public_processing_entry_point_writes_only_required_families(self):
        from json_processing import process
        old = processor.INSTITUTIONS
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw = root / 'fake.json'; raw.write_text(json.dumps([record('SYN-ONLY', [author(1, [A])])]))
                config = root / 'institutions.json'; config.write_text(json.dumps(INSTITUTIONS))
                before = (raw.read_bytes(), config.read_bytes())
                dest = root / 'processed'
                with redirect_stdout(StringIO()):
                    process(Namespace(input_json=str(raw), year=2025, years=None,
                        institutions_file=str(config), output=str(dest)))
                self.assertEqual({p.name for p in dest.iterdir()}, {'annual', 'processing_report.json'})
                self.assertEqual({p.name for p in (dest / 'annual').iterdir()},
                    {f'2025_{name}.csv' for name in CONTRACT})
                for name, spec in CONTRACT.items():
                    self.assertEqual(pd.read_csv(dest / 'annual' / f'2025_{name}.csv', nrows=0).columns.tolist(), spec['columns'])
                self.assertEqual((raw.read_bytes(), config.read_bytes()), before)
                self.assertFalse(json.loads((dest / 'processing_report.json').read_text())['scientific_comparison_performed'])
        finally:
            processor.INSTITUTIONS = old

    def test_missing_nonempty_required_field_is_not_silently_filled(self):
        tables = {name: pd.DataFrame(columns=spec['columns']) for name, spec in CONTRACT.items()}
        tables['author_affiliations'] = pd.DataFrame([{'eid': 'SYN-1'}])
        with self.assertRaisesRegex(ValueError, 'missing required columns'):
            project(tables)


if __name__ == '__main__':
    unittest.main()
