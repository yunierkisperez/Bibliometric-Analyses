"""Synthetic regression controls for several AF-IDs per focal institution.

These fixtures contain no downloaded records and do not validate manuscript
results. Unknown, nonfocal AF-IDs remain distinct under the existing method.
"""
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import scopus_process as processor
from plot_fig5_national_sharing import process_year
from plot_fig6_affiliation_structures import build_structures

A, A_ALIAS = "99000001", "99000004"
B, B_ALIAS = "99000002", "99000005"
OTHER = "99000999"
INSTITUTIONS = [
    {"afid": afid, "institution_id": institution, "name": name}
    for institution, name, afids in [
        ("EXAMPLE_01", "Fictitious University 01", (A, A_ALIAS)),
        ("EXAMPLE_02", "Fictitious University 02", (B, B_ALIAS)),
    ]
    for afid in afids
]


def author(identifier, afids):
    return {
        "@auid": str(identifier), "@seq": "1",
        "ce:indexed-name": f"Fictitious Author {identifier}",
        "ce:surname": f"Fictitious{identifier}", "ce:initials": "F",
        "affiliation": [{"@id": afid} for afid in afids],
    }


def record(eid, authors, corresponding_afid=None):
    afids = sorted({a["@id"] for person in authors for a in person["affiliation"]})
    head = {}
    if corresponding_afid is not None:
        head["correspondence"] = [
            {"affiliation": {"@afid": corresponding_afid}, "person": {}}
        ]
    return {"abstracts-retrieval-response": {
        "coredata": {
            "eid": eid, "dc:title": "Fictitious alias control",
            "prism:coverDate": "2025-01-01",
            "subtypeDescription": "Article", "subtype": "ar",
        },
        "authors": {"author": deepcopy(authors)},
        "affiliation": [
            {"@id": afid, "affilname": f"Fictitious affiliation {afid}",
             "affiliation-country": "Foreign" if afid == OTHER else "Ecuador"}
            for afid in afids
        ],
        "item": {"bibrecord": {"head": head}},
    }}


class InstitutionAliasControls(unittest.TestCase):
    def setUp(self):
        override = patch.object(processor, "INSTITUTIONS", deepcopy(INSTITUTIONS))
        override.start()
        self.addCleanup(override.stop)

    def compute(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2025_fictitious.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            before = path.read_bytes()
            tables = processor.process_input(path, analysis_year=2025, include_audit=True)
            self.assertEqual(path.read_bytes(), before)
            return tables

    def test_example_has_multiple_afids_and_one_mapping_row_per_institution(self):
        definitions = processor.load_institutions(ROOT / "config/institutions.example.json")
        self.assertEqual(len(definitions), 4)
        ids = {row["institution_id"] for row in definitions}
        self.assertEqual(len(ids), 3)
        aliases = [row for row in definitions if row["institution_id"] == "EXAMPLE_01"]
        self.assertEqual({row["afid"] for row in aliases}, {A, A_ALIAS})
        with (ROOT / "config/mapping.example.csv").open(encoding="utf-8-sig") as stream:
            mapping = list(csv.DictReader(stream))
        mapped = [row for row in mapping if row["institution_id"] != "NATIONAL"]
        self.assertEqual(len(mapped), len(ids))
        self.assertEqual({row["institution_id"] for row in mapped}, ids)
        self.assertEqual(
            {row["institution_id"]: row["name"] for row in mapped},
            {row["institution_id"]: row["name"] for row in definitions},
        )

    def test_conflicting_afid_or_alias_name_is_rejected(self):
        cases = [
            ([INSTITUTIONS[0], dict(INSTITUTIONS[2], afid=A)], "unique"),
            ([INSTITUTIONS[0], dict(INSTITUTIONS[1], name="Fictitious other name")], "same name"),
        ]
        for definitions, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "institutions.json"
                path.write_text(json.dumps(definitions), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    processor.load_institutions(path)

    def test_simultaneous_aliases_do_not_duplicate_counts_or_create_imfa(self):
        tables = self.compute([record("SYN-ALIAS", [author(1, [A, A_ALIAS])])])
        metrics = tables["institution_year_metrics"].set_index("institution_id")
        for institution in ("EXAMPLE_01", "NATIONAL"):
            self.assertEqual(int(metrics.loc[institution, "N_articles_attributed"]), 1)
            self.assertEqual(int(metrics.loc[institution, "N_authors_institution"]), 1)
            self.assertEqual(int(metrics.loc[institution, "N_author_article_signatures"]), 1)
        flags = tables["article_institution_flags"].set_index("institution_id")
        self.assertEqual(int(flags.loc["EXAMPLE_01", "flag_IMFA"]), 0)
        self.assertEqual(int(flags.loc["EXAMPLE_01", "flag_IMFA_raw_afid"]), 1)
        audit = tables["author_affiliation_canonicalization"].iloc[0]
        self.assertEqual(int(audit["n_distinct_raw_afids"]), 2)
        self.assertEqual(int(audit["n_distinct_canonical_affiliations"]), 1)
        self.assertEqual(set(audit["raw_afids"].split(";")), {A, A_ALIAS})
        correction = tables["affiliation_alias_correction_audit"].set_index("institution_id")
        self.assertEqual(int(correction.loc["EXAMPLE_01", "N_articles_IMFA_alias_only"]), 1)

    def test_alias_switch_is_not_cis_but_a_distinct_institution_is(self):
        records = [record("SYN-A", [author(1, [A])]),
                   record("SYN-A-ALIAS", [author(1, [A_ALIAS])])]
        tables = self.compute(records)
        focal = tables["article_institution_flags"].query("institution_id == 'EXAMPLE_01'")
        self.assertEqual(focal["flag_CIS"].tolist(), [0, 0])
        self.assertTrue(tables["cis_author_partner_institutions"].empty)
        tables = self.compute(records + [record("SYN-B", [author(1, [B_ALIAS])])])
        focal = tables["article_institution_flags"].query("institution_id == 'EXAMPLE_01'")
        self.assertEqual(focal["flag_CIS"].tolist(), [1, 1])

    def test_thirteen_papers_with_two_aliases_do_not_reach_hpa_threshold(self):
        tables = self.compute([
            record(f"SYN-P-{i}", [author(1, [A, A_ALIAS])]) for i in range(13)
        ])
        metrics = tables["institution_year_metrics"].set_index("institution_id")
        self.assertEqual(int(metrics.loc["EXAMPLE_01", "N_articles_attributed"]), 13)
        self.assertEqual(int(metrics.loc["EXAMPLE_01", "N_hyperprolific_authors"]), 0)
        productivity = tables["author_productivity"].query("institution_id == 'EXAMPLE_01'")
        self.assertEqual(productivity["p_author_institution_year"].tolist(), [13])
        top = tables["top_author_concentration"].query(
            "institution_id == 'EXAMPLE_01' and n_top_authors == 1"
        )
        self.assertEqual(top["n_articles_by_top_authors_union"].tolist(), [13])

    def test_twenty_five_papers_split_between_aliases_reach_hpa_threshold(self):
        tables = self.compute([
            record(f"SYN-P-{i}", [author(1, [A if i % 2 else A_ALIAS])])
            for i in range(25)
        ])
        metrics = tables["institution_year_metrics"].set_index("institution_id")
        self.assertEqual(int(metrics.loc["EXAMPLE_01", "N_hyperprolific_authors"]), 1)
        self.assertEqual(int(metrics.loc["EXAMPLE_01", "N_articles_with_hyperprolific_author"]), 25)
        productivity = tables["author_productivity"].query("institution_id == 'EXAMPLE_01'")
        self.assertEqual(productivity["p_author_institution_year"].tolist(), [25])

    def test_corresponding_author_at_either_alias_is_locally_led(self):
        for afid in (A, A_ALIAS):
            with self.subTest(afid=afid):
                tables = self.compute([
                    record("SYN-CA", [author(1, [A, A_ALIAS])], corresponding_afid=afid)
                ])
                flags = tables["article_institution_flags"].set_index("institution_id")
                for institution in ("EXAMPLE_01", "NATIONAL"):
                    self.assertEqual(int(flags.loc[institution, "CA_known"]), 1)
                    self.assertEqual(int(flags.loc[institution, "flag_CAL"]), 1)
                    self.assertEqual(int(flags.loc[institution, "flag_NLO"]), 0)

    def test_sharing_uses_canonical_institutions_instead_of_aliases(self):
        tables = self.compute([
            record("SYN-ALIAS", [author(1, [A, A_ALIAS])]),
            record("SYN-SHARED", [author(2, [A, A_ALIAS, B, B_ALIAS])]),
        ])
        national = process_year(2025, tables, Counter())["national"].iloc[0]
        self.assertEqual(int(national["total_unique_focal_attributed_articles"]), 2)
        self.assertEqual(int(national["articles_with_one_national_entity"]), 1)
        self.assertEqual(int(national["articles_with_two_or_more_national_entities"]), 1)
        self.assertEqual(int(national["distinct_national_entities"]), 2)
        self.assertEqual(float(national["shared_percent"]), 50.0)

    def test_partner_aliases_merge_by_union_for_imfa_and_cis(self):
        tables = self.compute([
            record("SYN-SHARED", [author(1, [A, A_ALIAS, B, B_ALIAS])]),
            record("SYN-B", [author(1, [B, B_ALIAS])]),
            record("SYN-B-ALIAS", [author(1, [B_ALIAS])]),
        ])
        mapping = pd.read_csv(ROOT / "config/mapping.example.csv", encoding="utf-8-sig")
        for table, column, eids in [
            ("imfa_author_partner_institutions", "imfa_article_eids", {"SYN-SHARED"}),
            ("cis_author_partner_institutions", "external_article_eids", {"SYN-B", "SYN-B-ALIAS"}),
        ]:
            with self.subTest(table=table):
                summary = build_structures(tables[table], column, mapping)["partner_summary"]
                focal = summary.query("institution_id == 'EXAMPLE_01'")
                self.assertEqual(len(focal), 1)
                partner = focal.iloc[0]
                self.assertEqual(partner["partner_key"], "FOCAL:EXAMPLE_02")
                self.assertEqual(set(partner["partner_afids_internal"].split(";")), {B, B_ALIAS})
                self.assertEqual(int(partner["n_articles"]), len(eids))
                self.assertEqual(int(partner["n_authors"]), 1)
                self.assertEqual(set(partner["article_eids_internal"].split(";")), eids)

    def test_primary_results_are_invariant_to_declared_alias_substitution(self):
        records = [
            record("SYN-1", [author(1, [A])]),
            record("SYN-2", [author(1, [A, B])], corresponding_afid=B),
            record("SYN-3", [author(1, [B])]),
            record("SYN-4", [author(2, [A]), author(3, [OTHER])]),
        ]
        changed = deepcopy(records)
        for index, item in enumerate(changed):
            response = item["abstracts-retrieval-response"]
            for person in response["authors"]["author"]:
                links = []
                for link in person["affiliation"]:
                    original = link["@id"]
                    alias = {A: A_ALIAS, B: B_ALIAS}.get(original, original)
                    links.append({"@id": alias})
                    if index % 2 == 0 and alias != original:
                        links.append({"@id": original})
                person["affiliation"] = links
            people = response["authors"]["author"]
            changed[index] = record(
                response["coredata"]["eid"], people,
                corresponding_afid=B_ALIAS if index == 1 else None,
            )
        original, aliased = self.compute(records), self.compute(changed)
        keys = {
            "institution_year_metrics": ["institution_id", "year"],
            "author_productivity": ["institution_id", "year", "author_id"],
            "top_author_concentration": ["institution_id", "year", "n_top_authors"],
        }
        for table, sort_columns in keys.items():
            with self.subTest(table=table):
                left = original[table].sort_values(sort_columns).reset_index(drop=True)
                right = aliased[table].sort_values(sort_columns).reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right)
        columns = ["eid", "institution_id", "flag_IMFA", "flag_CIS", "flag_HPA",
                   "CA_known", "flag_CAL", "flag_NLO"]
        left = original["article_institution_flags"][columns].sort_values(columns[:2])
        right = aliased["article_institution_flags"][columns].sort_values(columns[:2])
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
