"""Synthetic controls only; these tests do not validate manuscript results."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from table_export import SCHEMAS, read_table, write_xlsx
from table_legends import LEGENDS, LEGEND_SHEET, add_legend, legend_rows
from table_validation import validate_tables

S5_ROW = ['Expansion (2012-2018)', '2011-2018', 'IMFA', 0, 10, 0, 4, 0, 6, 10, 4, 6, None, 0.4, 0.4]


def worksheet_parts(path):
    with ZipFile(path) as package:
        return {name: package.read(name) for name in package.namelist() if name.startswith('xl/worksheets/')}


class LegendConfiguration(unittest.TestCase):
    def test_every_table_has_a_complete_legend(self):
        self.assertEqual(set(LEGENDS['tables']), {str(n) for n in range(1, 16)})
        for n in range(1, 16):
            spec = LEGENDS['tables'][str(n)]; headers = SCHEMAS[str(n)]['headers']
            covered = [i for group in spec['columns'] for i in group['columns']]
            self.assertEqual(covered, list(range(1, len(headers) + 1)), f'Table S{n}')
            self.assertTrue(spec['title'] and spec['description'])
            self.assertTrue(all(group['description'] for group in spec['columns']))
            self.assertTrue(set(spec['abbreviations']) <= set(LEGENDS['abbreviations']))
            self.assertTrue(set(spec['notes']) <= set(LEGENDS['notes']))
            self.assertIn('source', spec['notes'])
            rows = legend_rows(n, headers)
            self.assertEqual(rows[0][1], f'Table S{n}. {spec["title"]}')

    def test_legend_sheet_name_is_valid_and_distinct(self):
        self.assertLessEqual(len(LEGEND_SHEET), 31)
        self.assertFalse(set(LEGEND_SHEET) & set('[]:*?/\\'))
        self.assertNotIn(LEGEND_SHEET, [spec['sheet'] for spec in SCHEMAS.values()])

    def test_column_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'legend describes'):
            legend_rows(1, SCHEMAS['1']['headers'] + ['Unexpected'])


class LegendWorkbooks(unittest.TestCase):
    def test_exported_workbook_has_data_then_legend_and_is_readable(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'Table-S5.xlsx'
            write_xlsx(path, 5, [S5_ROW])
            formula = openpyxl.load_workbook(path, data_only=False); cached = openpyxl.load_workbook(path, data_only=True)
            self.assertEqual(formula.sheetnames, [SCHEMAS['5']['sheet'], LEGEND_SHEET])
            self.assertEqual(formula.active.title, SCHEMAS['5']['sheet'])
            self.assertEqual(formula.active['M2'].value, '=IF(D2=0,"",F2/D2)')
            self.assertEqual(cached.active['O2'].value, 0.4)
            legend = formula[LEGEND_SHEET]
            self.assertTrue(legend['A1'].value.startswith('Table S5. '))
            self.assertTrue(legend['A1'].font.b)
            self.assertTrue(legend['B3'].alignment.wrap_text)
            text = ' '.join(str(cell.value) for row in legend.iter_rows() for cell in row if cell.value)
            for header in SCHEMAS['5']['headers']:
                self.assertIn(' '.join(header.split()), text)
            self.assertIn('http://api.elsevier.com', text); self.assertIn('http://www.scopus.com', text)
            formula.close(); cached.close()
            frame = read_table(td, 5)
            self.assertEqual(len(frame), 1)

    def test_adding_a_legend_keeps_the_data_sheet_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            plain = Path(td) / 'plain' / 'Table-S5.xlsx'; plain.parent.mkdir()
            write_xlsx(plain, 5, [S5_ROW], legend=False)
            before = worksheet_parts(plain)
            self.assertEqual(len(before), 1)
            target = Path(td) / 'legend' / 'Table-S5.xlsx'
            result = add_legend(plain, target, 5)
            self.assertEqual(result['legend_sheet'], LEGEND_SHEET)
            after = worksheet_parts(target)
            self.assertEqual(len(after), 2)
            for name, data in before.items():
                self.assertEqual(after[name], data)
            self.assertEqual(worksheet_parts(plain), before)
            self.assertTrue(read_table(plain.parent, 5).equals(read_table(target.parent, 5)))
            with self.assertRaisesRegex(ValueError, 'already has'):
                add_legend(target, Path(td) / 'again.xlsx', 5)

    def test_reader_accepts_only_data_sheet_optionally_followed_by_legend(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'Table-S5.xlsx'
            write_xlsx(path, 5, [S5_ROW], legend=False)
            self.assertEqual(len(read_table(td, 5)), 1)
            book = openpyxl.load_workbook(path); book.create_sheet('Notes'); book.save(path)
            with self.assertRaisesRegex(ValueError, 'sheet name/order changed'):
                read_table(td, 5)
            write_xlsx(path, 5, [S5_ROW])
            book = openpyxl.load_workbook(path); book.move_sheet(LEGEND_SHEET, offset=-1); book.save(path)
            self.assertEqual(book.sheetnames, [LEGEND_SHEET, SCHEMAS['5']['sheet']])
            with self.assertRaisesRegex(ValueError, 'sheet name/order changed'):
                read_table(td, 5)

    def test_validation_ignores_the_legend_but_not_the_data(self):
        with tempfile.TemporaryDirectory() as td:
            reference = Path(td) / 'reference'; candidate = Path(td) / 'candidate'
            reference.mkdir(); candidate.mkdir()
            write_xlsx(reference / 'Table-S5.xlsx', 5, [S5_ROW], legend=False)
            write_xlsx(candidate / 'Table-S5.xlsx', 5, [S5_ROW])
            result = validate_tables(reference, candidate, tables=[5])[0]
            self.assertTrue(result['pass'])
            self.assertEqual(result['legend_sheet'], {'reference': False, 'candidate': True})
            changed = list(S5_ROW); changed[4] = 11
            write_xlsx(candidate / 'Table-S5.xlsx', 5, [changed])
            self.assertFalse(validate_tables(reference, candidate, tables=[5])[0]['pass'])

    def test_command_line_tool_writes_copies_and_refuses_overwrites(self):
        from add_table_legends import run
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'source'; source.mkdir(); output = Path(td) / 'output'
            write_xlsx(source / 'Table-S5.xlsx', 5, [S5_ROW], legend=False)
            original = (source / 'Table-S5.xlsx').read_bytes()
            report = run(source, output, ['S5'])
            self.assertEqual([(r['table'], r['data_sheet_unchanged']) for r in report], [(5, True)])
            self.assertEqual((source / 'Table-S5.xlsx').read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(source, output, ['S5'])
            with self.assertRaisesRegex(ValueError, 'different folder'):
                run(source, source, ['S5'])
            shutil.rmtree(output)
            with self.assertRaisesRegex(ValueError, 'Missing workbook'):
                run(source, output, ['S5', 'S6'])


if __name__ == '__main__':
    unittest.main()
