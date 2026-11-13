"""Presentation shared by the manuscript figures; never changes table values."""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONFIG=Path(__file__).resolve().parents[1]/'config'
PALETTE=json.loads((CONFIG/'palette.json').read_text())


def heatmap(axis, values, cmap=None, norm=None, aspect='auto', interpolation=None, **kwargs):
    """Editable vector rectangles with the same cell coordinates as imshow."""
    values=np.ma.masked_invalid(np.ma.asarray(values,dtype=float))
    rows,cols=values.shape
    mesh=axis.pcolormesh(np.arange(cols+1)-0.5,np.arange(rows+1)-0.5,values,
                         cmap=cmap,norm=norm,shading='flat',rasterized=False,**kwargs)
    axis.set_xlim(-0.5,cols-0.5);axis.set_ylim(rows-0.5,-0.5)
    axis.set_aspect(aspect)
    return mesh


def figure_key(path):
    stem=Path(path).stem
    aliases={'Fig3_supplementary_institution_heatmap':'Fig-S4',
             'Fig3_supplementary_national_annual_indicators':'Fig-S3',
             'Fig3_supplementary_institution_year_changes':'Fig-S5',
             'Figure_6_affiliation_partner_concentration_robust_D':'Fig6',
             'Figure_S9_affiliation_structures_2025_EC_CIS':'Fig-S9',
             'Figure_S10_robust_D_recurrence_partner_concentration':'Fig-S10'}
    return aliases.get(stem,stem)


def footer_legends(fig, national=True, lower=True, fontsize=10.0):
    for old in list(fig.legends):old.remove()
    handles=[]
    if national:handles.append(Line2D([0],[0],color='#111111',lw=2,label='National aggregate'))
    handles.extend([
        Line2D([0],[0],marker='o',ls='None',mfc='#888888',mec='white',ms=4.5,label='Upper robust signal (D ≥ 2)'),
        Line2D([0],[0],marker='o',ls='None',mfc='#888888',mec='#111111',ms=4.5,label='Black outline: |D| ≥ 3')])
    if lower:handles.append(Line2D([0],[0],marker='v',ls='None',mfc='#888888',mec='white',ms=4.5,label='Lower robust deviation (D ≤ −2)'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.5,0.065),ncol=len(handles),
               frameon=False,fontsize=fontsize,handlelength=1.5,handletextpad=0.4,columnspacing=1.0,borderaxespad=0)
    codes=sorted(c for c in PALETTE if c.startswith('U-'))
    # Matplotlib fills columns; this permutation gives U-01..U-15 on row one.
    ordered=[c for i in range(15) for c in codes[i::15]]
    handles=[Line2D([0],[0],marker='o',ls='None',color=PALETTE[c],mec='none',ms=7,label=c) for c in ordered]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.5,-0.004),ncol=15,
               frameon=False,fontsize=fontsize,handlelength=0.65,handletextpad=0.18,columnspacing=0.45,
               labelspacing=0.25,borderaxespad=0)


def panel_titles(axes):
    for i,ax in enumerate(axes):
        title=ax.get_title() or ax.get_title(loc='left')
        title=re.sub(r'^\([a-z]\)\s*','',title)
        for t in list(ax.texts):
            if re.fullmatch(r'\([a-z]\)',t.get_text()):t.remove()
        ax.set_title('');ax.set_title(f'({chr(97+i)}) {title}',loc='left',fontweight='bold',fontsize=9.3,pad=7)


def finish(fig, path):
    if getattr(fig,'_repro_finished',False):return
    fig._repro_finished=True
    key=figure_key(path)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    for ax in fig.axes:
        for obj in ax.get_children():
            if hasattr(obj,'set_rasterized'):obj.set_rasterized(False)
    if key in ['Fig1','Fig2','Fig4','Fig5','Fig6','Fig-S1']:
        footer_legends(fig,national=key!='Fig6',lower=key!='Fig4',fontsize=10.0)
    if key=='Fig1':
        footer_legends(fig,fontsize=10.5)
        panel_titles(fig.axes[:3])
        fig.subplots_adjust(bottom=0.155)
    elif key=='Fig2':
        fig.set_size_inches(7.35,8.0)
        panel_titles(fig.axes[:6])
        for text in list(fig.texts):
            if text.get_text()=='Year':text.remove()
        fig.subplots_adjust(left=0.09,right=0.99,top=0.96,bottom=0.155,wspace=0.25,hspace=0.48)
    elif key=='Fig4':
        panel_titles(fig.axes[:4])
        for text in list(fig.texts):
            if text.get_text()=='Year':text.remove()
        fig.set_size_inches(7.35,5.25)
        fig.subplots_adjust(left=0.105,right=0.99,top=0.95,bottom=0.21,wspace=0.30,hspace=0.48)
    elif key=='Fig5':
        panel_titles(fig.axes[:3]);fig.set_size_inches(7.35,6.6)
        fig.subplots_adjust(left=0.09,right=0.99,bottom=0.18,top=0.96,wspace=0.29,hspace=0.58)
    elif key=='Fig6':
        if fig._suptitle is not None:fig._suptitle.remove();fig._suptitle=None
        for ax in fig.axes[:4]:
            for t in list(ax.texts):
                if t.get_text()=='n' or t.get_text().isdigit():t.remove()
            ax.set_xticks([1,3,5,7,9,11,13,15]);ax.set_xticklabels([str(y) for y in range(2011,2026,2)])
        fig.subplots_adjust(top=0.955,bottom=0.20,left=0.09,right=0.985,hspace=0.28,wspace=0.12)
    elif key=='Fig7':
        for t in list(fig.texts):t.remove()
        fig._suptitle=None
        fig.axes[0].set_title('(a) Institution-year signal span',loc='left',fontsize=9,fontweight='bold',pad=28)
        fig.axes[1].set_title('(b) Full-period profile by indicator',loc='left',fontsize=9,fontweight='bold',pad=28)
        fig.subplots_adjust(top=0.90)
    elif key=='Fig-S1':
        fig.subplots_adjust(top=0.88,bottom=0.28)
    elif key=='Fig-S4':
        for ax in fig.axes[:2]:
            labels=[re.split(r'\s*[·:]\s*',x.get_text())[0] for x in ax.get_yticklabels()]
            ax.set_yticklabels(labels)
    elif key=='Fig-S9':
        if len(fig.axes)>=4:fig.delaxes(fig.axes[3])
        for old in list(fig.legends):old.remove()
        handles=[Line2D([0],[0],marker=marker,ls='None',color='#666666',ms=7,label=label)
                 for marker,label in [('o','Top-1'),('s','Top-5 union'),('^','Single recurrent partner')]]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(0.745,0.185),ncol=1,frameon=False,
                   fontsize=8,handletextpad=0.8,labelspacing=1.0)
        for text in list(fig.texts):
            if 'Panel c includes' in text.get_text():
                text.set_text('Panel c includes author–focal-university records with at least two observed nonfocal publications and a resolved partner affiliation.')
                text.set_fontsize(6.2);text.set_position((0.5,0.008));text.set_ha('center')
    elif key=='Fig-S10':
        fig.axes[0].set_title('Recurrence of robust-D partner-concentration signals',fontsize=9.2,pad=22)
    from vector_composition import tag_figure
    tag_figure(fig,key,path)
