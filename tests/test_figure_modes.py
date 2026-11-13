"""Check deliverable boundaries and cleanup for figures-only calculations."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_workflow import reproduce


class FigureModeControls(unittest.TestCase):
    def args(self, root, derived=False):
        return argparse.Namespace(
            figures_only=True, only_tables=None, minimal_tables=False,
            tables_dir=None, calculated_dir=None if derived else str(root/'calculated'),
            derived_dir=str(root/'annual') if derived else None,
            mapping_file=str(root/'fictitious_mapping.csv') if derived else None,
            mapping_encoding='utf-8-sig', work_dir=None, jobs=1,
            output=str(root/'result'), source_format='xlsx')

    def test_calculated_figures_only_keeps_no_tables_and_cleans_construction_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'calculated').mkdir(); construction = []
            def build(source, output, **kwargs):
                output.mkdir(parents=True)
                (output/'Table-S1.xlsx').write_bytes(b'fictitious construction data')
                construction.append(output)
                return {'fictitious': True}
            def render(tables, output, source_format, figures):
                self.assertTrue((tables/'Table-S1.xlsx').is_file())
                self.assertEqual(figures, ['Fig1'])
                (output/'Fig1.svg').write_text('<svg/>')
            with patch('runtime_check.check'), patch('table_export.build', side_effect=build), patch('figure_rendering.render', side_effect=render):
                reproduce('Fig1', self.args(root))
            self.assertFalse((root/'result/tables').exists())
            self.assertTrue((root/'result/figures/Fig1.svg').is_file())
            self.assertTrue(all(not p.exists() for p in construction))
            report = json.loads((root/'result/reproduction_report.json').read_text())
            self.assertEqual(report['selected_table_products'], [])
            self.assertTrue(report['table_provenance']['construction_tables_temporary'])

    def test_table_source_defaults_to_figure_only_with_only_its_required_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); tables = root/'article_subset'; tables.mkdir()
            (tables/'Table-S5.xlsx').write_bytes(b'fictitious input for a controlled rendering stub')
            args = self.args(root)
            args.tables_dir = str(tables); args.calculated_dir = None
            args.figures_only = False
            def render(source, output, source_format, figures):
                self.assertEqual(source, tables)
                self.assertEqual(figures, ['Fig3'])
                self.assertEqual({p.name for p in source.iterdir()}, {'Table-S5.xlsx'})
                (output/'Fig3.svg').write_text('<svg/>')
            with patch('runtime_check.check'), patch('table_export.build', side_effect=AssertionError('Existing tables must not be re-exported')), patch('figure_rendering.render', side_effect=render):
                reproduce('Fig3', args)
            self.assertFalse((root/'result/tables').exists())
            report = json.loads((root/'result/reproduction_report.json').read_text())
            self.assertEqual(report['selected_table_inputs'], ['S5'])
            self.assertEqual(report['selected_table_products'], [])

    def test_annual_figures_only_cleans_private_calculations_after_render_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'annual').mkdir(); private = []; construction = []
            def calculate(source, mapping, work, *args):
                work.mkdir(parents=True); private.append(work)
                return {'fictitious': True}
            def build(source, output, **kwargs):
                output.mkdir(parents=True); construction.append(output)
                (output/'Table-S1.xlsx').write_bytes(b'fictitious construction data')
                return {'fictitious': True}
            with patch('runtime_check.check'), patch('indicator_calculation.calculate', side_effect=calculate), patch('table_export.build', side_effect=build), patch('figure_rendering.render', side_effect=RuntimeError('controlled rendering failure')):
                with self.assertRaisesRegex(RuntimeError, 'controlled rendering failure'):
                    reproduce('Fig1', self.args(root, derived=True))
            self.assertFalse((root/'result/tables').exists())
            self.assertTrue(all(not p.exists() for p in private+construction))


if __name__ == '__main__':
    unittest.main()
