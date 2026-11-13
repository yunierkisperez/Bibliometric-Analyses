"""Construct the 15 manuscript table schemas from newly computed CSV values.

No reference workbook or reference scientific value is used for calculation.
XLSX serialization uses the Python standard library, including cached results
for the six original expressions per row in Tables S5 and S6.
"""
from pathlib import Path
import csv
import json
import math
import numbers
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCHEMAS = json.loads((ROOT/'config/table_schemas.json').read_text())
ADAPTERS = json.loads((ROOT/'config/table_adapters.json').read_text())


def letter(index):
    result = ''
    while index:
        index, rem = divmod(index-1, 26)
        result = chr(65+rem) + result
    return result


def value(v):
    if pd.isna(v) or v == '':
        return None
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        if not math.isfinite(float(v)):
            raise ValueError('Nonfinite table value')
        return int(v) if float(v).is_integer() else float(v)
    return str(v)


def formula(n, col, row):
    if n == 5:
        return {9:f'E{row}-D{row}',10:f'G{row}-F{row}',11:f'I{row}-H{row}',
                12:f'IF(D{row}=0,"",F{row}/D{row})',13:f'IF(E{row}=0,"",G{row}/E{row})',
                14:f'IF(J{row}=0,"",K{row}/J{row})'}.get(col)
    if n == 6:
        return {10:f'F{row}-E{row}',11:f'H{row}-G{row}',12:f'J{row}-I{row}',
                13:f'IF(E{row}=0,"",G{row}/E{row})',14:f'IF(F{row}=0,"",H{row}/F{row})',
                15:f'IF(K{row}=0,"",L{row}/K{row})'}.get(col)


def write_xlsx(path, n, rows):
    spec = SCHEMAS[str(n)]
    formats = list(dict.fromkeys(spec['formats']))
    numeric_formats = ''.join(f'<numFmt numFmtId="{164+i}" formatCode="{escape(f, {chr(34): "&quot;"})}"/>' for i, f in enumerate(formats))
    xfs = '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    xfs += '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"><alignment wrapText="1" vertical="center"/></xf>'
    xfs += ''.join(f'<xf numFmtId="{164+i}" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>' for i in range(len(formats)))
    styles = f'<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="{len(formats)}">{numeric_formats}</numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="{len(formats)+2}">{xfs}</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'
    sheet_rows = []
    for r, vals in enumerate([spec['headers'], *rows], 1):
        cells = []
        for c, raw in enumerate(vals):
            v = value(raw); coord=f'{letter(c+1)}{r}'
            style = 1 if r == 1 else 2 + formats.index(spec['formats'][c])
            attrs = f'r="{coord}" s="{style}"'
            f = formula(n,c,r) if r > 1 else None
            if f:
                # Values are freshly computed by the scientific generator.
                cached = '' if v is None else str(v)
                kind = ' t="str"' if v is None else ''
                cells.append(f'<c {attrs}{kind}><f>{escape(f)}</f><v>{cached}</v></c>')
            elif v is None:
                cells.append(f'<c {attrs}/>')
            elif isinstance(v, (int,float)):
                cells.append(f'<c {attrs}><v>{v!r}</v></c>')
            else:
                cells.append(f'<c {attrs} t="inlineStr"><is><t xml:space="preserve">{escape(v)}</t></is></c>')
        height = ' ht="60" customHeight="1"' if r == 1 else ''
        sheet_rows.append(f'<row r="{r}"{height}>{"".join(cells)}</row>')
    dim=f'A1:{letter(len(spec["headers"]))}{len(rows)+1}'
    filt=f'<autoFilter ref="{dim}"/>' if spec['auto_filter'] else ''
    freeze = ''
    if spec['freeze']:
        import re
        m=re.fullmatch(r'([A-Z]+)([0-9]+)',spec['freeze']);x=0
        for ch in m[1]:x=x*26+ord(ch)-64
        x-=1;y=int(m[2])-1
        freeze=f'<sheetViews><sheetView workbookViewId="0"><pane xSplit="{x}" ySplit="{y}" topLeftCell="{spec["freeze"]}" state="frozen"/></sheetView></sheetViews>'
    cols=''.join(f'<col min="{i+1}" max="{i+1}" width="{max(14,min(38,max(map(len,str(h).splitlines()))+2))}" customWidth="1"/>' for i,h in enumerate(spec['headers']))
    worksheet=f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="{dim}"/>{freeze}<cols>{cols}</cols><sheetData>{"".join(sheet_rows)}</sheetData>{filt}</worksheet>'
    parts={
        '[Content_Types].xml':'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
        '_rels/.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        'xl/workbook.xml':f'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{escape(spec["sheet"], {chr(34): "&quot;"})}" sheetId="1" r:id="rId1"/></sheets><calcPr calcId="0" fullCalcOnLoad="1"/></workbook>',
        'xl/_rels/workbook.xml.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        'xl/styles.xml':styles,'xl/worksheets/sheet1.xml':worksheet,
    }
    with ZipFile(path,'w',compression=ZIP_DEFLATED) as z:
        for name, text in parts.items():z.writestr(name,'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+text)


def build(generated, output, tables=None, include_main=None):
    generated=Path(generated);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    selected=list(range(1,16)) if tables is None else sorted(set(tables))
    if any(n not in range(1,16) for n in selected):raise ValueError('Table numbers must be within 1-15')
    include_main=tables is None if include_main is None else include_main
    manifest=[]
    for n in selected:
        spec=ADAPTERS[str(n)];frames=[]
        for name in spec['sources']:
            found=list(generated.rglob(name))
            if len(found)!=1:raise ValueError(f'Missing or ambiguous calculated table: {name}')
            frames.append(pd.read_csv(found[0],keep_default_na=False,low_memory=False))
        if n==7:
            nat=frames[1].rename(columns={'national_output':'output','national_authors':'institution_authors',**{f'national_CTA_{k}':f'CTA_{k}' for k in [1,5,10]}})
            nat['id-paper']='National';frames=[frames[0],nat]
        df=pd.concat(frames,ignore_index=True)
        if n==2:
            source=list(generated.rglob('Source-Fig-S1.csv'))
            if len(source)!=1:raise ValueError('Expanded Table S2 requires the calculated IMFA-or-EC-CIS union; calculate --tables S2 first')
            union=pd.read_csv(source[0],keep_default_na=False,float_precision='round_trip').rename(columns={
                'N_articles_IMFA_or_CIS':'count','N_articles_attributed':'denominator','IMFA_or_CIS_rate':'value_raw'})
            union['metric_key']='IMFA_or_CIS'
            union['denominator_column']='N_articles_attributed'
            df=pd.concat([df,union],ignore_index=True)
            if df.duplicated(['id-paper','analysis_year','metric_key']).any():
                raise ValueError('Table S2 contains duplicate institution-year-indicator keys')
        if n==4:df=df.loc[df['unit_type'].eq('national')].copy()
        if n in [1,7]:df=df.sort_values(['id-paper','year' if n==1 else 'analysis_year'],kind='stable')
        if n==2:
            metric_order=['IMFA','CIS','HPA','HAO','NLO','HPA_MFA_or_CIS_articles','IMFA_or_CIS']
            df['_metric_order']=pd.Categorical(df['metric_key'],metric_order,ordered=True)
            # The manuscript orders the pooled aggregate after the focal codes.
            df['_national']=df['id-paper'].eq('National')
            df=df.sort_values(['_metric_order','_national','id-paper','analysis_year'],kind='stable')
        if n==3:df=df.sort_values(['id-paper','metric_label'],kind='stable')
        if n==6:
            df=df.loc[df['id-paper'].ne('National')].sort_values(['id-paper','phase','metric_label','display_period'],kind='stable')
        if n==12:df=df.sort_values(['year','institution_code','shared_articles','partner_code'],ascending=[True,True,False,True],kind='stable')
        rows=[]
        for _,row in df.iterrows():
            vals=[]
            for ci,field in enumerate(spec['fields']):
                if field['field'] not in df.columns:raise ValueError(f'Table S{n}: missing {field["field"]}')
                v=row[field['field']]
                if field.get('divide') and v!='':v=float(v)/field['divide']
                if field.get('translate'):v=field['translate'].get(str(v),v)
                if field.get('text_bool'):v='True' if str(v).lower()=='true' else 'False'
                if SCHEMAS[str(n)]['kinds'][ci]=='number' and v!='' and pd.notna(v):v=float(v)
                if SCHEMAS[str(n)]['kinds'][ci]=='mixed' and str(v).isdigit():v=int(v)
                vals.append(value(v))
            rows.append(vals)
        schema=SCHEMAS[str(n)]
        with (output/f'Table-S{n}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(schema['headers']);writer.writerows(rows)
        write_xlsx(output/schema['filename'],n,rows)
        manifest.append({'table':n,'rows':len(rows),'columns':len(schema['headers'])})
    if include_main:
        if 6 in selected:
            main=write_main_table(output,output/'Main-Table1.csv','xlsx')
        else:
            import tempfile
            with tempfile.TemporaryDirectory(prefix='repro_main_table_') as tmp:
                build(generated,tmp,tables=[6],include_main=False)
                main=write_main_table(tmp,output/'Main-Table1.csv','xlsx')
        manifest.append(main)
    return manifest


def read_table(folder, n, source_format='xlsx'):
    folder=Path(folder);spec=SCHEMAS[str(n)]
    if source_format=='xlsx':
        import openpyxl
        wb=openpyxl.load_workbook(folder/spec['filename'],data_only=True,read_only=True)
        if wb.sheetnames!=[spec['sheet']]:raise ValueError(f'Table S{n}: sheet name/order changed')
        rows=list(wb.active.values);wb.close()
        if not rows or list(rows[0])!=spec['headers']:raise ValueError(f'Table S{n}: columns changed')
        df=pd.DataFrame(rows[1:],columns=spec['headers'])
    else:
        df=pd.read_csv(folder/f'Table-S{n}.csv',keep_default_na=False,low_memory=False,float_precision='round_trip')
        if list(df.columns)!=spec['headers']:raise ValueError(f'Table S{n}: columns changed')
    df.columns=[x['field'] for x in ADAPTERS[str(n)]['fields']]
    for ci,key in enumerate(df.columns):
        if spec['kinds'][ci]=='number':df[key]=pd.to_numeric(df[key].replace('',float('nan')),errors='raise')
    for field in ADAPTERS[str(n)]['fields']:
        key=field['field']
        if field.get('divide'):df[key]=pd.to_numeric(df[key],errors='coerce')*field['divide']
        if field.get('translate'):
            reverse={v:k for k,v in field['translate'].items()};df[key]=df[key].replace(reverse)
        if field.get('text_bool'):
            text=df[key].map(lambda x:str(x).lower())
            if not text.isin(['true','false']).all():raise ValueError(f'Table S{n}: invalid or missing Boolean flag')
            df[key]=text.eq('true')
    return df


def write_main_table(folder, output, source_format='xlsx'):
    """Recalculate representative annual signals in manuscript Table 1.

    The institution presentation order follows the paper. Signal selection,
    largest-absolute-D representatives and values are calculated from S6.
    """
    data=read_table(folder,6,source_format)
    primary=['At least one, excluding NLO','IMFA','CIS','HPA','HAO','HPA-MFA/CIS']
    selected=data.loc[data['display_period'].eq('2024-2025')&data['metric_label'].isin(primary)&data['robust_change_D'].abs().ge(2)].copy()
    selected['_absD']=selected['robust_change_D'].abs()
    selected['_metric_order']=selected['metric_label'].map({v:i for i,v in enumerate(primary)})
    representatives=selected.sort_values(['_absD','_metric_order'],ascending=[False,True],kind='stable').drop_duplicates('id-paper')
    paper_order=['U-17','U-10','U-24','U-27','U-15','U-12','U-09']
    order={code:i for i,code in enumerate(paper_order)}
    representatives['_order']=representatives['id-paper'].map(order).fillna(len(order))
    representatives=representatives.sort_values(['_order','id-paper'],kind='stable')
    headers=['Institution','Metric','ΔN','ΔE','ΔU','Rate, 2024 → 2025 (%)','D','Signal','Strict']
    rows=[]
    def signed(v):
        if int(v)!=v:raise ValueError('A manuscript table count is not an integer')
        return f'{int(v):+d}'.replace('-','−')
    if data.duplicated(['id-paper','metric_label','display_period']).any():raise ValueError('Duplicate manuscript Table 1 source key')
    for _,row in representatives.iterrows():
        code=row['id-paper'];metric=row['metric_label']
        n=row['net_output_change'];e=row['indicator_positive_change'];u=row['remaining_output_change']
        d=row['robust_change_D'];direction='upper' if d>=2 else 'lower' if d<=-2 else ''
        signal=('Strong' if abs(d)>=3 else 'Elevated')+' '+direction if direction else 'No signal' if pd.notna(d) else 'Not evaluable'
        strict=(n>0 and e>0 and u<=0) or (n<0 and e<0 and u>=0)
        rows.append([code,'EC-CIS' if metric=='CIS' else metric,signed(n),signed(e),signed(u),
            f'{row["indicator_output_share_start_pct"]:.1f}→{row["indicator_output_share_end_pct"]:.1f}',
            f'{d:+.2f}'.replace('-','−') if pd.notna(d) else 'NA',signal,'Yes' if strict else 'No'])
    with Path(output).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
    return {'product':'Main-Table1.csv','source':'Table S6','rows':len(rows),'columns':len(headers),
        'source_signal_count':len(selected),'signal_institution_count':int(selected['id-paper'].nunique()),
        'strong_signal_count':int(selected['_absD'].ge(3).sum())}
