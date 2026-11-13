"""Synthetic controls for independent inputs, selected outputs and safe failures."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from indicator_calculation import annual_inputs
from product_catalog import analysis_selection, figure_products
from json_processing import json_jobs
from table_export import build, read_table
from table_validation import validate_tables


class ModularControls(unittest.TestCase):
    def test_one_selected_year_does_not_require_other_jsons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'one.json').write_text('[]')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'inputs': {'2011': 'one.json', '2025': 'not_downloaded.json'}}))
            self.assertEqual(list(json_jobs(manifest, years=[2011])), [2011])

    def test_duplicate_selected_year_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'one.json').write_text('[]')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'inputs': {'2011': 'one.json'}}))
            with self.assertRaisesRegex(ValueError, 'duplicates'):
                json_jobs(manifest, years=[2011, 2011])

    def test_growth_can_start_with_only_its_fifteen_annual_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for year in range(2011, 2026):
                (root / f'{year}_institution_year_metrics.csv').write_text('year\n')
            self.assertEqual(len(annual_inputs(root, ['growth'])), 15)
            duplicate = root / 'duplicate'; duplicate.mkdir()
            (duplicate / '2011_institution_year_metrics.csv').write_text('year\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                annual_inputs(root, ['growth'])

    def test_one_table_export_does_not_require_other_calculations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); calculated = root / 'calculated'; calculated.mkdir()
            source = calculated / 'fig1_data_used.csv'
            with source.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id-paper', 'year', 'N_articles_attributed', 'annual_growth_pct',
                                 'growth_D', 'growth_deviation_side', 'growth_deviation_strength'])
                writer.writerow(['U-01', 2011, 10, '', '', '', ''])
            output = root / 'tables'
            build(calculated, output, tables=[1], include_main=False)
            self.assertEqual({p.name for p in output.iterdir()}, {'Table-S1.xlsx', 'Table-S1.csv'})
            self.assertEqual(read_table(output, 1).iloc[0]['N_articles_attributed'], 10)
            reference = root / 'reference'; reference.mkdir()
            shutil.copyfile(output / 'Table-S1.xlsx', reference / 'Table-S1.xlsx')
            result = validate_tables(reference, output, tables=[1])
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]['pass'])

    def test_unrelated_table_cannot_be_requested_from_figure_entry_point(self):
        with self.assertRaisesRegex(ValueError, 'supports table products'):
            figure_products('Fig1', only_tables=['S13'])

    def test_supplementary_heatmap_does_not_pull_main_figure_calculations(self):
        figures, products = figure_products('Fig-S8')
        self.assertEqual(figures, ['Fig-S8'])
        self.assertEqual(analysis_selection(tables=products), ['sharing'])
        self.assertEqual(figure_products('Fig3', minimal_tables=True)[1], ['S5'])


if __name__ == '__main__':
    unittest.main()
