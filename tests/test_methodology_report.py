"""Controlled serialization checks for the private methodology report."""
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('methodology_report_under_test',ROOT/'src/output_utils.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MethodologyReportControls(unittest.TestCase):
    def test_counts_precision_and_undefined_values_are_not_lost(self):
        p_value=1.2345678901234567e-20
        frame=pd.DataFrame([{'n':7,'rho':0.1234567890123456,'p_value':p_value}, {'n':0,'rho':np.nan,'p_value':np.nan}])
        stream=StringIO()
        with redirect_stdout(stream):
            module.report_methodology(fictitious=frame,count=np.int64(7),undefined=float('nan'))
        line=stream.getvalue().strip()
        self.assertTrue(line.startswith('METHOD_DIAGNOSTICS_JSON '))
        value=json.loads(line.removeprefix('METHOD_DIAGNOSTICS_JSON '))
        self.assertEqual(value['count'],7)
        self.assertEqual(value['fictitious'][0]['n'],7)
        self.assertEqual(value['fictitious'][0]['rho'],0.1234567890123456)
        self.assertEqual(value['fictitious'][0]['p_value'],p_value)
        self.assertIsNone(value['fictitious'][1]['rho'])
        self.assertIsNone(value['undefined'])


if __name__=='__main__':
    unittest.main()
