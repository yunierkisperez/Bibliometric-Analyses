"""Synthetic controls only; these tests do not validate manuscript results."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import scopus_process as processor
from table_export import SCHEMAS, read_table, write_xlsx
from json_processing import json_jobs

A='99000001';A_ALIAS='99000002';B='99000003';OTHER='99000099'
INSTITUTIONS=[{'afid':A,'institution_id':'FAKE_A','name':'Fictitious University A'},
              {'afid':A_ALIAS,'institution_id':'FAKE_A','name':'Fictitious University A'},
              {'afid':B,'institution_id':'FAKE_B','name':'Fictitious University B'}]


def author(aid,afids):
    return {'@auid':str(aid),'@seq':'1','ce:indexed-name':f'Fictitious Author {aid}',
            'ce:surname':f'Fictitious{aid}','ce:initials':'F',
            'affiliation':[{'@id':afid} for afid in afids]}


def record(eid,authors,cover='2025-01-01',subtype='Article'):
    afids=list(dict.fromkeys(af['@id'] for a in authors for af in a['affiliation']))
    return {'abstracts-retrieval-response':{
        'coredata':{'eid':eid,'dc:title':'Fictitious test record','prism:coverDate':cover,
                    'subtypeDescription':subtype,'subtype':'ar'},
        'authors':{'author':deepcopy(authors)},
        'affiliation':[{'@id':afid,'affilname':f'Fictitious affiliation {afid}',
                        'affiliation-country':'Foreign' if afid==OTHER else 'Ecuador'} for afid in afids],
        'item':{'bibrecord':{'head':{}}},
    }}


class ProcessingControls(unittest.TestCase):
    def setUp(self):
        self.old=processor.INSTITUTIONS;processor.INSTITUTIONS=deepcopy(INSTITUTIONS)

    def tearDown(self):processor.INSTITUTIONS=self.old

    def compute(self,records,year=2025):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'2025_fictitious.json';path.write_text(json.dumps(records))
            return processor.process_input(path,analysis_year=year,include_audit=True)

    def test_pubyear_overrides_cover_date(self):
        tables=self.compute([record('SYN-YEAR',[author(1,[A])],cover='2024-12-31')])
        self.assertEqual(set(tables['institution_year_metrics']['year']),{2025})
        self.assertEqual(int(tables['articles'].iloc[0]['publication_year']),2024)

    def test_focal_aliases_are_consolidated(self):
        tables=self.compute([record('SYN-ALIAS',[author(1,[A,A_ALIAS])])])
        flags=tables['article_institution_flags'].set_index('institution_id')
        self.assertEqual(int(flags.loc['FAKE_A','flag_IMFA']),0)
        self.assertEqual(int(flags.loc['FAKE_A','flag_IMFA_raw_afid']),1)

    def test_national_counts_unique_eids(self):
        tables=self.compute([record('SYN-SHARED',[author(1,[A,B])]),record('SYN-ONE',[author(2,[A])])])
        metrics=tables['institution_year_metrics'].set_index('institution_id')
        self.assertEqual(int(metrics.loc['NATIONAL','N_articles_attributed']),2)
        self.assertEqual(int(metrics.loc[['FAKE_A','FAKE_B'],'N_articles_attributed'].sum()),3)

    def test_imfa_requires_the_focal_authors_own_links(self):
        tables=self.compute([record('SYN-OWN',[author(1,[A]),author(2,[B,OTHER])])])
        flags=tables['article_institution_flags'].set_index('institution_id')
        self.assertEqual(int(flags.loc['FAKE_A','flag_IMFA']),0)
        self.assertEqual(int(flags.loc['FAKE_B','flag_IMFA']),1)

    def test_ec_cis_uses_a_distinct_nonfocal_publication(self):
        tables=self.compute([record('SYN-FOCAL',[author(1,[A]),author(2,[B])]),
                             record('SYN-ALTERNATIVE',[author(1,[OTHER]),author(3,[B])])])
        flags=tables['article_institution_flags'].set_index(['eid','institution_id'])
        self.assertEqual(int(flags.loc[('SYN-FOCAL','FAKE_A'),'flag_CIS']),1)

    def test_duplicate_and_excluded_document_types(self):
        one=record('SYN-ONE',[author(1,[A])])
        tables=self.compute([one,deepcopy(one),record('SYN-ERR',[author(2,[A])],subtype='Erratum')])
        self.assertEqual(set(tables['articles']['eid']),{'SYN-ONE'})

    def test_hpa_threshold_is_25(self):
        tables=self.compute([record(f'SYN-HPA-{i}',[author(1,[A])]) for i in range(25)])
        metrics=tables['institution_year_metrics'].set_index('institution_id')
        self.assertEqual(int(metrics.loc['FAKE_A','N_articles_with_hyperprolific_author']),25)

    def test_reserved_national_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'institutions.json';p.write_text(json.dumps([{'afid':A,'institution_id':'NATIONAL','name':'Fictitious'}]))
            with self.assertRaisesRegex(ValueError,'reserved'):processor.load_institutions(p)


class TableAndInputControls(unittest.TestCase):
    def test_formula_and_missing_cache_are_preserved(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'Table-S5.xlsx'
            row=['Expansion (2012-2018)','2011-2018','IMFA',0,10,0,4,0,6,10,4,6,None,0.4,0.4]
            write_xlsx(p,5,[row])
            formula=openpyxl.load_workbook(p,data_only=False);cached=openpyxl.load_workbook(p,data_only=True)
            self.assertEqual(formula.active['M2'].value,'=IF(D2=0,"",F2/D2)')
            self.assertIn(cached.active['M2'].value,[None,''])
            self.assertEqual(cached.active['O2'].value,0.4)
            formula.close();cached.close()

    def test_extra_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with (Path(td)/'Table-S1.csv').open('w',newline='') as f:
                csv.writer(f).writerow(SCHEMAS['1']['headers']+['Unexpected'])
            with self.assertRaisesRegex(ValueError,'columns changed'):read_table(td,1,'csv')

    def test_invalid_signal_flag_is_not_silently_false(self):
        with tempfile.TemporaryDirectory() as td:
            with (Path(td)/'Table-S15.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(SCHEMAS['15']['headers'])
                writer.writerow([2025,'U-01','IMFA',3.5,'unknown','True'])
            with self.assertRaisesRegex(ValueError,'Boolean flag'):read_table(td,15,'csv')

    def test_duplicate_manifest_year_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'inputs.json';p.write_text('{"inputs":{"2011":"a","2011":"b"}}')
            with self.assertRaisesRegex(ValueError,'Duplicate year'):json_jobs(p)

    def test_incomplete_period_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'inputs.json';p.write_text('{"inputs":{"2025":"fictitious.json"}}')
            with self.assertRaisesRegex(ValueError,'2011-2025'):json_jobs(p)


if __name__=='__main__':unittest.main()
