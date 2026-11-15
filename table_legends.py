"""Legend sheets for the supplementary Excel tables.

A legend is explanatory text only: the table title, a description, the meaning
of every column, the abbreviations used and general notes. It contains no data.
The legend is appended as a second sheet; the data sheet is not modified, so
formulas and their cached results are preserved byte for byte.
"""
from pathlib import Path
import json
import os
import re
from html import unescape
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
LEGENDS = json.loads((ROOT / 'config/table_legends.json').read_text(encoding='utf-8'))
LEGEND_SHEET = LEGENDS['sheet']
WORKSHEET_TYPE = 'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'
WORKSHEET_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet'
QUOTE = {'"': '&quot;'}


def data_sheets(names):
    """Sheet names of a workbook without the optional legend sheet."""
    return [name for name in names if name != LEGEND_SHEET]


def allowed_sheets(data_sheet):
    """The two accepted layouts: data sheet alone, or data sheet followed by the legend."""
    return [data_sheet], [data_sheet, LEGEND_SHEET]


def legend_rows(n, headers):
    """Rows of the legend sheet as (kind, first cell, second cell); None is an empty row.

    Kinds: title and label (bold first cell), head (both bold), entry (both wrapped).
    """
    spec = LEGENDS['tables'][str(n)]
    covered = [i for group in spec['columns'] for i in group['columns']]
    if covered != list(range(1, len(headers) + 1)):
        raise ValueError(f'Table S{n}: the legend describes {len(covered)} columns but the table has {len(headers)}')
    label = lambda text: ' '.join(str(text).split())
    rows = [('title', f'Table S{n}. {spec["title"]}', None), None,
            ('label', 'Description', spec['description']), None,
            ('head', 'Column', 'Description')]
    for group in spec['columns']:
        rows.append(('entry', '; '.join(label(headers[i - 1]) for i in group['columns']), group['description']))
    if spec['abbreviations']:
        rows += [None, ('head', 'Abbreviation', 'Meaning')]
        rows += [('entry', key, LEGENDS['abbreviations'][key]) for key in spec['abbreviations']]
    if spec['notes']:
        rows += [None, ('head', 'Note', 'Explanation')]
        rows += [('entry', LEGENDS['notes'][key]['label'], LEGENDS['notes'][key]['text']) for key in spec['notes']]
    return rows


def legend_xml(rows, bold, wrap):
    """Worksheet XML with inline strings; `bold` and `wrap` are cell-format indexes."""
    cell = lambda ref, style, text: (f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">'
                                     f'{escape(text)}</t></is></c>')
    body = []
    for number, row in enumerate(rows, 1):
        if row is None:
            continue
        kind, first, second = row
        left = wrap if kind == 'entry' else bold
        cells = cell(f'A{number}', left, first)
        if second is not None:
            cells += cell(f'B{number}', bold if kind == 'head' else wrap, second)
        body.append(f'<row r="{number}">{cells}</row>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:B{len(rows)}"/><sheetViews><sheetView workbookViewId="0"/></sheetViews>'
            '<sheetFormatPr defaultRowHeight="15"/>'
            '<cols><col min="1" max="1" width="52" customWidth="1"/><col min="2" max="2" width="120" customWidth="1"/></cols>'
            f'<sheetData>{"".join(body)}</sheetData></worksheet>')


def _counted(xml, tag, child):
    """Locate <tag count="N">...</tag> and check that N matches its children."""
    match = re.search(rf'<{tag}\b([^>]*)>(.*?)</{tag}>', xml, re.S)
    if not match:
        raise ValueError(f'Workbook styles have no <{tag}> block')
    count = re.search(r'\bcount="(\d+)"', match.group(1))
    children = len(re.findall(rf'<{child}[\s/>]', match.group(2)))
    if not count or int(count.group(1)) != children:
        raise ValueError(f'Workbook styles: <{tag}> count does not match its entries')
    return match, children


def _append(xml, match, tag, addition, total):
    start, end = match.span()
    attributes = re.sub(r'\bcount="\d+"', f'count="{total}"', match.group(1), count=1)
    return xml[:start] + f'<{tag}{attributes}>{match.group(2)}{addition}</{tag}>' + xml[end:]


def _styles(xml):
    """Append one bold font and two cell formats; return (xml, bold index, wrap index)."""
    match, fonts = _counted(xml, 'fonts', 'font')
    first = re.search(r'<font\b[^>]*/>|<font\b[^>]*>.*?</font>', match.group(2), re.S).group(0)
    if first.endswith('/>'):
        bold_font = '<font><b/></font>'
    else:
        bold_font = re.sub(r'^(<font\b[^>]*>)(?:<b\s*/>)?', r'\1<b/>', first, count=1)
    xml = _append(xml, match, 'fonts', bold_font, fonts + 1)
    match, formats = _counted(xml, 'cellXfs', 'xf')
    base = ' xfId="0"' if '<cellStyleXfs' in xml else ''
    addition = (f'<xf numFmtId="0" fontId="{fonts}" fillId="0" borderId="0"{base} applyFont="1" applyAlignment="1">'
                '<alignment vertical="top"/></xf>'
                f'<xf numFmtId="0" fontId="0" fillId="0" borderId="0"{base} applyAlignment="1">'
                '<alignment vertical="top" wrapText="1"/></xf>')
    return _append(xml, match, 'cellXfs', addition, formats + 2), formats, formats + 1


def _document_properties(xml):
    """Keep the optional sheet list of docProps/app.xml in step; leave unknown layouts untouched."""
    pair_pattern = (r'(<HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>[^<]*</vt:lpstr>'
                    r'</vt:variant><vt:variant><vt:i4>)(\d+)(</vt:i4></vt:variant></vt:vector></HeadingPairs>)')
    title_pattern = r'(<TitlesOfParts><vt:vector size=")(\d+)(" baseType="lpstr">)(.*?)(</vt:vector></TitlesOfParts>)'
    pairs = re.search(pair_pattern, xml); titles = re.search(title_pattern, xml, re.S)
    if not pairs or not titles or pairs.group(2) != titles.group(2):
        return xml
    total = int(titles.group(2)) + 1
    xml = re.sub(title_pattern, lambda m: (f'{m.group(1)}{total}{m.group(3)}{m.group(4)}'
                                           f'<vt:lpstr>{escape(LEGEND_SHEET)}</vt:lpstr>{m.group(5)}'), xml, count=1, flags=re.S)
    return re.sub(pair_pattern, lambda m: f'{m.group(1)}{total}{m.group(3)}', xml, count=1)


def add_legend(source, target, n, headers=None):
    """Write a copy of workbook `source` to `target` with the legend of Table S{n} as second sheet.

    Only package metadata, the style list and the new sheet are written; every
    other part, including the data sheet, keeps its exact bytes. `source` and
    `target` may be the same path.
    """
    source = Path(source); target = Path(target)
    with ZipFile(source) as package:
        infos = package.infolist()
        parts = {info.filename: package.read(info.filename) for info in infos}
    workbook = parts['xl/workbook.xml'].decode('utf-8')
    sheets = re.findall(r'<sheet\b[^>]*/>', workbook)
    names = [unescape(re.search(r'\bname="([^"]*)"', sheet).group(1)) for sheet in sheets]
    if LEGEND_SHEET in names:
        raise ValueError(f'Table S{n}: the workbook already has a "{LEGEND_SHEET}" sheet')
    if len(sheets) != 1:
        raise ValueError(f'Table S{n}: expected one data sheet, found {len(sheets)}')
    if headers is None:
        import openpyxl
        book = openpyxl.load_workbook(source, data_only=True, read_only=True)
        headers = list(next(book[names[0]].iter_rows(min_row=1, max_row=1, values_only=True)))
        book.close()
    rows = legend_rows(n, headers)
    styles, bold, wrap = _styles(parts['xl/styles.xml'].decode('utf-8'))
    prefix = re.search(r'\b(\w+):id="', sheets[0]).group(1)
    relations = parts['xl/_rels/workbook.xml.rels'].decode('utf-8')
    used = set(re.findall(r'\bId="([^"]+)"', relations))
    relation = next(f'rId{i}' for i in range(1, len(used) + 2) if f'rId{i}' not in used)
    index = next(i for i in range(1, len(parts) + 2) if f'xl/worksheets/sheet{i}.xml' not in parts)
    part = f'xl/worksheets/sheet{index}.xml'
    sheet_id = max(int(x) for x in re.findall(r'\bsheetId="(\d+)"', workbook)) + 1
    entry = f'<sheet name="{escape(LEGEND_SHEET, QUOTE)}" sheetId="{sheet_id}" {prefix}:id="{relation}"/>'
    types = parts['[Content_Types].xml'].decode('utf-8')
    for text, closing in [(workbook, '</sheets>'), (relations, '</Relationships>'), (types, '</Types>')]:
        if text.count(closing) != 1:
            raise ValueError(f'Table S{n}: unexpected workbook package layout')
    changed = {
        'xl/workbook.xml': workbook.replace('</sheets>', entry + '</sheets>'),
        'xl/_rels/workbook.xml.rels': relations.replace(
            '</Relationships>', f'<Relationship Id="{relation}" Type="{WORKSHEET_REL}" Target="worksheets/sheet{index}.xml"/></Relationships>'),
        '[Content_Types].xml': types.replace(
            '</Types>', f'<Override PartName="/{part}" ContentType="{WORKSHEET_TYPE}"/></Types>'),
        'xl/styles.xml': styles,
    }
    if 'docProps/app.xml' in parts:
        changed['docProps/app.xml'] = _document_properties(parts['docProps/app.xml'].decode('utf-8'))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.legend.tmp')
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as package:
            for info in infos:
                data = changed[info.filename].encode('utf-8') if info.filename in changed else parts[info.filename]
                copy = ZipInfo(info.filename, date_time=info.date_time)
                copy.compress_type = ZIP_DEFLATED; copy.external_attr = info.external_attr
                package.writestr(copy, data)
            new = ZipInfo(part, date_time=infos[0].date_time); new.compress_type = ZIP_DEFLATED
            new.external_attr = infos[0].external_attr
            package.writestr(new, legend_xml(rows, bold, wrap).encode('utf-8'))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return {'table': n, 'data_sheet': names[0], 'legend_sheet': LEGEND_SHEET, 'legend_rows': len(rows)}
