# Data access and private configuration

## Figures from supplementary tables

Obtain Tables S1–S15 from the article's supplementary material. They are not
included in this code repository. Each figure requires only the files listed
in its script's `--help` and in [README.md](README.md).

This workflow reads anonymous institution codes. It requires no Scopus account,
raw records, API key or institution-to-code mapping.

## Calculations from Scopus JSONs

The software reads locally downloaded Abstract Retrieval JSON records. Obtain
data under your own authorized Scopus access and retain their retrieval dates.
The scripts do not retrieve data or store API keys.

Three private configuration files define annual JSON paths, focal institution
AF-IDs and the display mapping. [USAGE.md](USAGE.md) explains their formats,
including how to assign several AF-IDs to one institution. Files in `config/`
are fictitious examples, not the study's confidential assignments.

To match the article from JSONs, use the study's data snapshot, parameters and
authorized institution/code assignments. The confidential mapping is not
distributed. Without it, researchers can apply the methods to their own
institutions but cannot identify the anonymous study institutions from this
repository. Reproducing figures from the supplementary tables avoids that
requirement.

Table S9 sequence identifiers are assigned to ranked calculated rows. They
are calculated outputs, not imported labels; their values can change when the
underlying records or ranking change.

## Handling and redistribution

Keep JSON records, credentials, private configurations, annual CSVs, calculation
files and logs outside the public code directory. Some intermediates contain
author or affiliation identifiers. Coded identifiers do not by themselves
establish that a dataset is anonymous or may be redistributed.

The software licence covers code and documentation, not Scopus data. Article
supplements and separate dataset deposits have their own access and distribution
conditions. Check the conditions applicable to your data before sharing results
or records in a repository.

For scholarly outputs, acknowledge Scopus and record the actual retrieval dates.
An attribution statement can use this form, with both placeholders completed:

> Data were downloaded from the Scopus API between [start date] and [end date]
> via http://api.elsevier.com and http://www.scopus.com.
