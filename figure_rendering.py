"""Render the manuscript inventory using only anonymous supplementary tables.

Each family runs in a separate Python process so Matplotlib settings cannot
leak between figures. No private mapping, annual record CSV or JSON is read.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'src'
sys.path.insert(0,str(SRC))
from table_export import read_table
from product_catalog import FIGURES, figure_selection


def metadata():
    palette=json.loads((ROOT/'config/palette.json').read_text())
    return pd.DataFrame([{'id-paper':code,'institution_id':'NATIONAL' if code=='National' else code,
        'institution_code':code,'name':code,'color_hex':color,'institution_color':color,
        'code_order':int(code.split('-')[-1]) if code.startswith('U-') else 0,
        'legend_label':code,'line_width':3.2 if code=='National' else 2.0} for code,color in palette.items()])


def attach(df, code='id-paper'):
    df=df.copy()
    if code!='id-paper':df['id-paper']=df[code]
    extra=metadata().drop(columns=[x for x in metadata().columns if x!='id-paper' and x in df.columns])
    result=df.merge(extra,on='id-paper',how='left',validate='many_to_one')
    if result['color_hex'].isna().any():raise ValueError('Unknown anonymous institution code')
    return result


def numeric(df, columns):
    for col in columns:df[col]=pd.to_numeric(df[col].replace('',np.nan),errors='raise')
    return df


def family(number, folder, output, source_format, figures=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from font_resources import configure_matplotlib, FAMILY
    configure_matplotlib()
    plt.rcParams.update({'font.family':FAMILY,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'font.size':8,'svg.hashsalt':'bibliometrics-reproduction-1'})
    os.environ.pop('REPRO_DATA_ONLY',None)
    table=lambda n:read_table(folder,n,source_format)
    mapping=metadata()
    wanted=set(figures if figures is not None else [key for key,spec in FIGURES.items() if spec[0]==number])
    if not wanted or any(key not in FIGURES or FIGURES[key][0]!=number for key in wanted):
        raise ValueError('Selected figures do not belong to this rendering family')
    if number==1:
        import plot_fig1_national_growth_robust_d as m
        data=attach(table(1))
        data['growth_status']=np.where(data['annual_growth_pct'].notna(),'computed_consecutive_years','first_analysis_year')
        data['growth_strong_deviation']=data['growth_D'].abs().ge(3)
        signals=data.loc[data['growth_deviation_side'].isin(['upper','lower'])].copy()
        m.build_combined_figure(data,signals,mapping,output/'Fig1',2011,2025,7.5,7.6,300,0.58)
    elif number==2:
        import plot_fig2_temporal_exposure_profiles as m
        source=table(2)
        data=attach(source.loc[source['metric_key'].isin([x['key'] for x in m.METRICS])])
        data['value_plot']=100*data['value_raw']
        data['eligible']=data['institution_id'].ne('NATIONAL') & data['denominator'].ge(50) & data['value_plot'].notna()
        data['metric_order']=data['metric_key'].map({x['key']:i for i,x in enumerate(m.METRICS,1)})
        data['extreme_deviation']=data['robust_D'].abs().ge(3)
        distribution,_=m.classify_annual_distributions(data,5,8,2,3)
        signals=data.loc[data['deviation_side'].isin(['upper','lower'])].copy()
        m.build_figure(data,distribution,signals,mapping,output/'Fig2',2011,2025,0.58,2,7.35,8.8,300)
    elif number==3:
        import plot_fig3_compositional_change as m
        def accounting(n):
            raw=table(n)
            if n==5:raw['id-paper']='National'
            data=attach(raw);data['unit_type']='national' if n==5 else 'institution'
            lookup={v:k for k,v in m.ACCOUNTING_LABELS.items()};data['metric_key']=data['metric_label'].map(lookup)
            years=data['display_period'].str.extract(r'^(\d{4})-(\d{4})$').astype(int)
            data['start_year']=years[0];data['end_year']=years[1]
            data['interval_type']=np.where(data['end_year']-data['start_year']>1,'phase_endpoint','annual_transition')
            data['indicator_output_share_change_pp']=data['indicator_output_share_end_pct']-data['indicator_output_share_start_pct']
            data['both_endpoint_outputs_positive']=data['output_start'].gt(0)&data['output_end'].gt(0)
            data['strict_indicator_supported_growth']=data['net_output_change'].gt(0)&data['indicator_positive_change'].gt(0)&data['remaining_output_change'].le(0)
            data['strict_indicator_concentrated_contraction']=data['net_output_change'].lt(0)&data['indicator_positive_change'].lt(0)&data['remaining_output_change'].ge(0)
            if n==6:
                data['robust_change_strong_signal']=data['robust_change_D'].abs().ge(3)
                data['robust_change_value_pp']=data['indicator_rate_change_pp']
            return data
        frames=[]
        if wanted.intersection(['Fig3','Fig-S3']):frames.append(accounting(5))
        if wanted.intersection(['Fig-S4','Fig-S5']):frames.append(accounting(6))
        data=pd.concat(frames,ignore_index=True)
        phase=data.loc[data['interval_type'].eq('phase_endpoint')]
        annual=data.loc[data['interval_type'].eq('annual_transition')]
        if 'Fig3' in wanted:m.create_main_figure(phase,annual,output,'Fig3',300)
        if 'Fig-S4' in wanted:m.create_accounting_heatmap(phase.loc[phase['unit_type'].eq('institution')],mapping,output,'Fig3',300)
        if 'Fig-S3' in wanted:m.create_national_annual_indicator_heatmap(annual,output,'Fig3',300)
        if 'Fig-S5' in wanted:m.create_institution_transition_scatterplots(annual,output,'Fig3',300)
    elif number==4:
        import plot_fig4_author_concentration as m
        if 'Fig4' in wanted:
            data=attach(table(7));national=data.loc[data['institution_id'].eq('NATIONAL')].copy()
            national=national.rename(columns={f'CTA_{k}':f'national_CTA_{k}' for k in [1,5,10]})
            data=data.loc[data['institution_id'].ne('NATIONAL')].copy();signals=[]
            for field,label,_,_,_ in m.PLOT_METRICS:
                dcol={'A_top50':'D_A_top50'}.get(field,f'D_{field}')
                selected=data.loc[data[dcol].ge(2)].copy();selected['metric']=label
                selected['value']=selected[field];selected['signal_level']=np.where(selected[dcol].ge(3),'extreme','elevated')
                signals.append(selected)
            signals=pd.concat(signals,ignore_index=True)
            m.plot_main_figure(data,national,signals,50,5,2011,2025,2,output/'Fig4',300)
        if 'Fig-S6' in wanted:m.plot_cumulative_profiles(attach(table(8)),output/'Fig-S6',300)
    elif number==5:
        import plot_fig5_national_sharing as m
        if wanted.intersection(['Fig5','Fig-S7']):
            institution=attach(table(11),'institution_code')
            signal_frames=[]
            for key,dcol in [('shared_percent','shared_D'),('shared_NLO_percent','shared_NLO_D')]:
                direction=np.where(institution[dcol].ge(2),'upper',np.where(institution[dcol].le(-2),'lower',''))
                level=np.where(institution[dcol].abs().ge(3),'strong',np.where(institution[dcol].abs().ge(2),'elevated',''))
                institution[dcol+'_signal_direction']=direction;institution[dcol+'_signal_level']=level
                part=institution.loc[institution[dcol].abs().ge(2)].copy();part['metric_column']=key
                part['value_percent']=part[key];part['signal_direction']=direction[institution[dcol].abs().ge(2)]
                part['signal_level']=level[institution[dcol].abs().ge(2)];signal_frames.append(part)
            signals=pd.concat(signal_frames,ignore_index=True)
        if 'Fig5' in wanted:
            national=table(10)
            for key in m.MECHANISMS:
                national[key+'_percent_of_shared']=100*national[key+'_articles']/national['articles_with_two_or_more_national_entities'].replace(0,np.nan)
            distribution=m.make_distribution_table(institution,50)
            m.plot_main_figure(national,institution,distribution,signals,50,m.MIN_SHARED_CA_KNOWN_FOR_FILLED_POINT,output/'Fig5')
        if 'Fig-S7' in wanted:m.plot_supplementary_top_partners(institution,50,output/'Fig-S7')
        if 'Fig-S8' in wanted:
            pairs=table(12)
            pairs['dependency_percent']=100*pairs['shared_articles']/pairs['focal_total_articles'].replace(0,np.nan)
            m.plot_supplementary_matrix(pairs,output/'Fig-S8')
    elif number==6:
        import plot_fig6_affiliation_structures as m
        annual=attach(table(13),'institution_code')
        rename={x:x.replace('ec_cis_','cis_') for x in annual if x.startswith('ec_cis_')}
        rename['n_ec_cis_observed_nonfocal_publications']='n_cis_external_articles'
        annual=annual.rename(columns=rename)
        focal=annual['institution_code'].ne('National')
        annual['eligible_imfa_structure']=(focal&annual['n_articles_attributed'].ge(50)&annual['n_imfa_articles'].ge(10)).astype(int)
        annual['eligible_cis_structure']=(focal&annual['n_articles_attributed'].ge(50)&annual['n_cis_external_articles'].ge(10)).astype(int)
        annual['eligible_joint_structure']=(annual['eligible_imfa_structure'].eq(1)&annual['eligible_cis_structure'].eq(1)).astype(int)
        dframes=[]
        for signal,measure,eligible,col,denom in m.TEMPORAL_METRICS:
            key=m.METRIC_KEY_LOOKUP[(signal,measure)];stored=table(13).set_index(['year','institution_code'])[key+'_D']
            part=annual.loc[annual[eligible].eq(1)].copy();part['signal']=signal;part['measure']=measure;part['metric_key']=key;part['value']=part[col]
            part['D']=[stored.loc[(r.year,r.institution_code)] for r in part.itertuples(index=False)]
            part['reference_evaluable']=part['D'].notna().astype(int)
            for c,condition in [('upper_elevated_D_ge_2',part['D'].ge(2)),('upper_extreme_D_ge_3',part['D'].ge(3)),('lower_deviation_D_le_minus_2',part['D'].le(-2)),('lower_extreme_D_le_minus_3',part['D'].le(-3))]:part[c]=condition.astype(int)
            dframes.append(part)
        dvalues=pd.concat(dframes,ignore_index=True)
        recurrence=m.robust_d_recurrence_table(dvalues)
        if 'Fig6' in wanted:m.make_temporal_figure(annual,dvalues,output,2011,2025,300)
        if 'Fig-S10' in wanted:m.make_recurrence_figure(recurrence,annual['institution_code'].unique().tolist(),output,2011,2025,300)
        if 'Fig-S9' in wanted:
            summary=annual.loc[annual['year'].eq(2025)].merge(table(14),on=['year','institution_code'],how='left',validate='one_to_one')
            m.make_supplementary_2025_figure(summary,pd.DataFrame(columns=['institution_code']),output,2025,300)
    elif number==7:
        import plot_fig7_exposure_signal_matrix as m
        details=attach(table(15),'institution_code');details['scorable']=details['robust_deviation'].notna()
        details['metric_order']=details['metric_key'].map({k:i for i,k in enumerate(m.METRIC_ORDER,1)})
        cells,row_order=m.build_cell_summary(details,mapping,2011,2025,2023)
        profiles=m.build_institution_metric_profiles(details,row_order)
        references=details.groupby(['analysis_year','metric_key'],as_index=False)['scorable'].any().rename(columns={'scorable':'reference_valid'})
        m.plot_matrix(cells,row_order,profiles,references,output/'Fig7',2011,2025,2023,2,3,300)
    elif number==8:
        import plot_supplementary_temporal_deviations as m
        if 'Fig-S1' in wanted:
            from supplementary_inputs import imfa_or_cis_from_s2
            data=attach(imfa_or_cis_from_s2(table(2)))
            data['value_plot']=100*data['IMFA_or_CIS_rate']
            distribution,signals=m.classify_imfa_or_cis(data,50,5,8,2,3)
            m.plot_imfa_or_cis(data,distribution,signals,mapping,output/'Fig-S1',2011,2025,50,2,300)
        if 'Fig-S2' in wanted:
            summary=attach(table(3));lookup={v.replace('\n',' '):k for k,v in m.HEATMAP_LABELS.items()}
            summary['metric_key']=summary['metric_label'].map(lookup)
            order=summary.groupby(['institution_id','code_order'],as_index=False).agg(maximum_metric_deviation_years=('deviation_years','max'),total_metric_deviation_years=('deviation_years','sum')).sort_values(['maximum_metric_deviation_years','total_metric_deviation_years','code_order'],ascending=[False,False,True])['institution_id'].tolist()
            dmatrix=summary.pivot(index='institution_id',columns='metric_key',values='median_robust_D').reindex(index=order,columns=m.MAIN_METRICS)
            cmatrix=summary.pivot(index='institution_id',columns='metric_key',values='deviation_years').reindex(index=order,columns=m.MAIN_METRICS)
            m.plot_persistent_heatmap(summary,dmatrix,cmatrix,order,output/'Fig-S2',2,300)
    else:raise ValueError('Unknown figure family')


RENAMES={
    'Fig3_supplementary_institution_heatmap':'Fig-S4',
    'Fig3_supplementary_national_annual_indicators':'Fig-S3',
    'Fig3_supplementary_institution_year_changes':'Fig-S5',
    'Figure_6_affiliation_partner_concentration_robust_D':'Fig6',
    'Figure_S9_affiliation_structures_2025_EC_CIS':'Fig-S9',
    'Figure_S10_robust_D_recurrence_partner_concentration':'Fig-S10',
}


def render(folder, output, source_format, figures=None):
    output=Path(output)
    from vector_composition import compose
    from table_export import SCHEMAS
    expected=figure_selection(figures)
    products=list(dict.fromkeys(t for key in expected for t in FIGURES[key][1]))
    inputs=[Path(folder)/(SCHEMAS[key[1:]]['filename'] if source_format=='xlsx' else f'Table-{key}.csv') for key in products]
    def hashes():
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    before=hashes();composition=[]
    env=dict(os.environ,PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    env.pop('REPRO_DATA_ONLY',None)
    with tempfile.TemporaryDirectory(prefix='repro_figures_') as tmp:
        work=Path(tmp)
        for n in sorted({FIGURES[key][0] for key in expected}):
            dest=work/str(n);dest.mkdir()
            log=output/f'family_{n}.log'
            with log.open('w') as f:
                proc=subprocess.run([sys.executable,__file__,'--family',str(n),'--tables-dir',str(folder),
                    '--output',str(dest),'--source-format',source_format,'--figures',
                    *[key for key in expected if FIGURES[key][0]==n]],stdout=f,stderr=subprocess.STDOUT,env=env)
            if proc.returncode:raise RuntimeError(f'Figure family {n} failed; see {log}')
            for meta in sorted(dest.glob('*.layout.json')):
                stem=meta.name.removesuffix('.layout.json')
                key=RENAMES.get(stem,stem)
                if key not in expected:raise ValueError(f'An unrequested figure was rendered: {key}')
                composition.append(compose(dest/(stem+'.svg'),meta,output/key))
            print(f'Rendered figure family {n}',flush=True)
    missing=[stem+ext for stem in expected for ext in ['.pdf','.svg'] if not (output/(stem+ext)).is_file()]
    if missing:raise ValueError(f'Missing vector figures: {missing}')
    report=[]
    import xml.etree.ElementTree as ET
    import fitz
    for stem in expected:
        tree=ET.parse(output/(stem+'.svg'))
        images=tree.findall('.//{http://www.w3.org/2000/svg}image')
        texts=ET.parse(output/(stem+'_text.svg')).findall('.//{http://www.w3.org/2000/svg}text')
        with fitz.open(output/(stem+'.pdf')) as pdf:
            pdf_images=sum(len(page.get_images()) for page in pdf)
            size=list(pdf[0].rect)[2:]
        report.append({'figure':stem,'svg_embedded_images':len(images),'pdf_embedded_images':pdf_images,
                       'live_svg_text_nodes':len(texts),'pdf_page_size_pt':size,
                       'vector_structure_pass':not images and pdf_images==0 and bool(texts),
                       'visual_identity_tested_by_this_check':False})
    (output/'vector_report.json').write_text(json.dumps(report,indent=2))
    if before!=hashes():raise ValueError('An anonymous table changed during rendering')
    (output/'figure_build_report.json').write_text(json.dumps({'source_format':source_format,
        'selected_figures':expected,'input_sha256':before,'inputs_unchanged':True,'composition':composition,
        'research_json_or_private_mapping_read':False},ensure_ascii=False,indent=2))
    if not all(x['vector_structure_pass'] for x in report):raise ValueError('A figure contains raster content or lacks its live-text companion')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',type=int,required=True)
    p.add_argument('--tables-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-format',choices=['xlsx','csv'],default='xlsx')
    p.add_argument('--figures',nargs='+');a=p.parse_args()
    family(a.family,a.tables_dir,a.output,a.source_format,a.figures)
