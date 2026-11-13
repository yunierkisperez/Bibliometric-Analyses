"""Controls for the original, data-derived sequence-row numbering rule."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sequence_ids_under_test', ROOT / 'src/output_utils.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SequenceIdControls(unittest.TestCase):
    def test_sorted_result_gets_the_original_sequential_labels_without_files(self):
        frame = pd.DataFrame([
            {'id-paper': 'FICTITIOUS-02', 'start_year': 2011, 'end_year': 2015},
            {'id-paper': 'FICTITIOUS-01', 'start_year': 2012, 'end_year': 2014},
        ])
        before = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'CONFIG', Path(temp)):
            with patch.dict(os.environ, {'REPRO_SEQUENCE_ID_REGISTRY': '/nonexistent/old_registry.json'}):
                self.assertEqual(module.persistence_ids(frame), ['EP-001', 'EP-002'])
        pd.testing.assert_frame_equal(frame, before)

    def test_distinct_sequences_with_the_same_endpoints_receive_distinct_ids(self):
        frame = pd.DataFrame([
            {'id-paper': 'FICTITIOUS-01', 'start_year': 2011, 'end_year': 2013},
            {'id-paper': 'FICTITIOUS-01', 'start_year': 2011, 'end_year': 2013},
        ])
        self.assertEqual(module.persistence_ids(frame), ['EP-001', 'EP-002'])

    def test_empty_result_has_no_identifiers(self):
        self.assertEqual(module.persistence_ids(pd.DataFrame()), [])


if __name__ == '__main__':
    unittest.main()
