"""Compose newly calculated vector marks with data-free manuscript designs.

The design files contain axes, fixed labels and typography. Scientific marks
and numerical annotations are taken exclusively from the current Matplotlib
rendering and the current tables. No reference data plot is copied.
"""
from copy import deepcopy
from pathlib import Path
import json
import math
import re
import xml.etree.ElementTree as ET
import numpy as np
from font_resources import FAMILY, load_glyphs, character_positions

ROOT=Path(__file__).resolve().parent
SVG='http://www.w3.org/2000/svg'
XLINK='http://www.w3.org/1999/xlink'
NS={'s':SVG}
ET.register_namespace('',SVG);ET.register_namespace('xlink',XLINK)


def tag_figure(fig, key, path):
    """Label data artists and record semantic anchors before serialization."""
    axes=list(fig.axes)
    for ax in list(axes):axes.extend(a for a in ax.child_axes if a not in axes)
    design_path=ROOT/'designs/layout.json'
    if design_path.is_file():
        design=json.loads(design_path.read_text()).get(key)
        if design:
            if len(axes)!=len(design['axes']):raise ValueError('Figure axes no longer match the manuscript design')
            # The fixed tick labels in the design must retain their data scale
            # even when an input perturbation changes Matplotlib autoscaling.
            for ax,frame in zip(axes,design['axes']):
                ax.set_xlim(frame['baseline_xlim']);ax.set_ylim(frame['baseline_ylim'])
    fig.canvas.draw()
    record={'figure':key,'axes':[]}
    for i,ax in enumerate(axes):
        ax.set_gid(f'repro_axis_{i}')
        bbox=ax.get_window_extent()
        row={'index':i,'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
             'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),
             'xticklabels':[t.get_text() for t in ax.get_xticklabels()],
             'yticklabels':[t.get_text() for t in ax.get_yticklabels()],
             'ytick_anchors':[1-(float(ax.transData.transform((0,v))[1])-bbox.y0)/bbox.height for v in ax.get_yticks()],
             'data_ids':[],'data_roles':{},'labels':[]}
        for j,obj in enumerate([*ax.lines,*ax.patches,*ax.collections,*ax.images]):
            gid=f'repro_data_{i}_{j}';obj.set_gid(gid);row['data_ids'].append(gid)
            row['data_roles'][gid]=type(obj).__name__
        for j,text in enumerate(ax.texts):
            s=text.get_text()
            if not (re.match(r'^(?:[-+−]?\d|U-\d|N-\d|NA\b|--)',s)):
                continue
            if re.fullmatch(r'\([a-z]\)',s):continue
            gid=f'repro_label_{i}_{j}';text.set_gid(gid)
            anchor=text.get_transform().transform(text.get_position())
            label={'gid':gid,'text':s,'ha':text.get_ha(),'va':text.get_va(),
                   'fontweight':str(text.get_fontweight()),'fontsize':float(text.get_fontsize()),
                   'anchor':[(float(anchor[0])-bbox.x0)/bbox.width,1-(float(anchor[1])-bbox.y0)/bbox.height],
                   'color':text.get_color()}
            if hasattr(text,'xy'):
                point=ax.transData.transform(text.xy)
                label['point_anchor']=[(float(point[0])-bbox.x0)/bbox.width,1-(float(point[1])-bbox.y0)/bbox.height]
            row['labels'].append(label)
        record['axes'].append(row)
    Path(str(path)+'.layout.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))


def style_dict(text):
    return dict((x.split(':',1)[0].strip(),x.split(':',1)[1].strip()) for x in text.split(';') if ':' in x)


def canonical_color(color):
    from matplotlib.colors import to_hex
    return to_hex(color,keep_alpha=False)


def face_for(element, inherited=None):
    attrs=dict(inherited or {});attrs.update(element.attrib)
    attrs.update(style_dict(element.get('style','')))
    if attrs.get('data-repro-face'):return attrs['data-repro-face']
    weight=str(attrs.get('font-weight','normal')).lower()
    return 'bold' if weight in ['bold','700','800','900'] else 'semibold' if weight in ['600','semibold'] else 'regular'


def outline_text(root, glyphs):
    """Convert live text to outlines using the bundled open font files."""
    parents={c:p for p in root.iter() for c in p}
    for text in list(root.iter('{'+SVG+'}text')):
        attrs=style_dict(text.get('style',''));attrs.update(text.attrib)
        face=face_for(text);size=float(re.match(r'[0-9.]+',attrs.get('font-size','10'))[0])
        group=ET.Element('{'+SVG+'}g',{k:v for k,v in text.attrib.items() if k in ['transform','fill','opacity','id']})
        group.set('data-label',''.join(text.itertext()))
        children=list(text) if list(text) else [text]
        for span in children:
            sf=face_for(span,attrs);spec=glyphs.get(sf,glyphs['regular'])
            ss=float(re.match(r'[0-9.]+',span.get('font-size',str(size)))[0])
            content=span.text or ''
            xs=[float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',span.get('x',text.get('x','0')),re.I)]
            ys=[float(y) for y in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',span.get('y',text.get('y','0')),re.I)]
            x=xs[0] if xs else 0;y=ys[0] if ys else 0
            for j,char in enumerate(content):
                if j<len(xs):x=xs[j]
                if j<len(ys):y=ys[j]
                item=spec.get(char) or glyphs['regular'].get(char)
                if item is None:
                    if char.isspace():x+=ss*0.25;continue
                    raise ValueError(f'No {FAMILY} glyph for {char!r}')
                if item['d']:
                    ET.SubElement(group,'{'+SVG+'}path',{'d':item['d'],'transform':f'matrix({ss} 0 0 {-ss} {x} {y})'})
                x+=ss*item['advance']
        parent=parents[text];position=list(parent).index(text);parent.remove(text);parent.insert(position,group)


def path_bounds(path):
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.svgLib.path import parse_path
    pen=BoundsPen(None);parse_path(path,pen)
    return pen.bounds


def axis_rect(element):
    # Matplotlib emits its axes background as the first patch group.
    for child in element:
        if child.get('id','').startswith('patch_'):
            path=child.find('s:path',NS)
            if path is not None:return path_bounds(path.get('d'))
    raise ValueError('Cannot locate axes background')


def transformed_group(group, source, target, clip_id):
    """Transform world coordinates while retaining physical marker dimensions."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.svgLib.path import parse_path
    sx=(target[2]-target[0])/(source[2]-source[0]);sy=(target[3]-target[1])/(source[3]-source[1])
    tx=target[0]-sx*source[0];ty=target[1]-sy*source[1]
    result=deepcopy(group)
    def walk(node,in_defs=False):
        in_defs=in_defs or node.tag=='{'+SVG+'}defs'
        if not in_defs:
            if 'clip-path' in node.attrib:node.set('clip-path',f'url(#{clip_id})')
            if node.tag=='{'+SVG+'}path' and node.get('d'):
                pen=SVGPathPen(None);parse_path(node.get('d'),TransformPen(pen,(sx,0,0,sy,tx,ty)));node.set('d',pen.getCommands())
            elif node.tag=='{'+SVG+'}use':
                node.set('x',str(float(node.get('x','0'))*sx+tx));node.set('y',str(float(node.get('y','0'))*sy+ty))
            elif node.tag=='{'+SVG+'}image':raise ValueError('A scientific layer unexpectedly contains raster data')
        for child in node:walk(child,in_defs)
    walk(result)
    return result


def smooth_colorbar(group,target,defs,gradient_id,clip_id):
    """Keep continuous colourbars as editable vector gradients, without seams."""
    w=target[2]-target[0];h=target[3]-target[1]
    if not ((w<25 and h>50) or (h<20 and w>50)):return group
    samples=[]
    for node in group.iter('{'+SVG+'}path'):
        fill=node.get('fill') or style_dict(node.get('style','')).get('fill')
        if fill and fill!='none' and node.get('d'):
            bounds=path_bounds(node.get('d'))
            samples.append(((bounds[0]+bounds[2])/2 if w>h else -(bounds[1]+bounds[3])/2,fill))
    if len(samples)<128:return group
    samples.sort(key=lambda v:v[0]);colors=[c for _,c in samples]
    attrs={'id':gradient_id,'gradientUnits':'userSpaceOnUse',
        'x1':str(target[0]),'y1':str(target[3] if h>w else target[1]),
        'x2':str(target[0] if h>w else target[2]),'y2':str(target[1])}
    gradient=ET.SubElement(defs,'{'+SVG+'}linearGradient',attrs)
    stops=[(0,colors[0])]+[((i+.5)/len(colors),c) for i,c in enumerate(colors)]+[(1,colors[-1])]
    for offset,color in stops:ET.SubElement(gradient,'{'+SVG+'}stop',{'offset':str(offset),'stop-color':color})
    result=ET.Element('{'+SVG+'}g',{'id':group.get('id','')})
    ET.SubElement(result,'{'+SVG+'}rect',{'x':str(target[0]),'y':str(target[1]),'width':str(w),'height':str(h),
        'fill':f'url(#{gradient_id})','clip-path':f'url(#{clip_id})'})
    return result


def portable_svg(root):
    """Serialize presentation attributes explicitly for vector editors."""
    for node in root.iter():
        for name,value in style_dict(node.attrib.pop('style','')).items():
            node.set(name,value)
    def opacity(node,inherited=1):
        alpha=inherited*float(node.attrib.pop('opacity','1'))
        if node.tag=='{'+SVG+'}g':
            for child in node:opacity(child,alpha)
        else:
            if alpha!=1:node.set('opacity',str(alpha))
            for child in node:opacity(child)
    # Illustrator's single-primitive opacity groups can make Cairo introduce
    # thin bitmap masks. Primitive opacity retains these as vector paths.
    opacity(root)


def add_text(parent,text,x,y,size,face='regular',fill='#222222'):
    e=ET.SubElement(parent,'{'+SVG+'}text',{'font-family':FAMILY,'font-size':str(size),'fill':fill})
    if face!='regular':e.set('font-weight','700' if face=='bold' else '600')
    positions=character_positions(text,x,size,face)
    ET.SubElement(e,'{'+SVG+'}tspan',{'x':' '.join(str(v) for v in positions),'y':str(y)}).text=text
    return e


def compose(source_svg, meta_path, output_base, text_mode='outlines'):
    key=json.loads(Path(meta_path).read_text())['figure']
    design_dir=ROOT/'designs';layout=json.loads((design_dir/'layout.json').read_text())[key]
    glyphs=load_glyphs()
    meta=json.loads(Path(meta_path).read_text());source=ET.parse(source_svg).getroot()
    root=ET.parse(design_dir/(key+'.svg')).getroot()
    defs=root.find('s:defs',NS)
    if defs is None:defs=ET.SubElement(root,'{'+SVG+'}defs')
    all_ids={e.get('id'):e for e in source.iter() if e.get('id')}
    drawing_order={e.get('id'):j for j,e in enumerate(source.iter()) if e.get('id')}
    slots={e.get('id'):e for e in root.iter() if e.get('id')}
    label_audit=[]
    for axis in meta['axes']:
        i=axis['index'];design=layout['axes'][i];target=design['rect']
        under=slots[f'repro_under_{i}'];layer=slots[f'repro_over_{i}']
        axis_node=all_ids.get(f'repro_axis_{i}')
        if axis_node is None:continue
        source_rect=axis_rect(axis_node)
        clip_id=f'repro_clip_{i}';clip=ET.SubElement(defs,'{'+SVG+'}clipPath',{'id':clip_id})
        ET.SubElement(clip,'{'+SVG+'}rect',{'x':str(target[0]),'y':str(target[1]),'width':str(target[2]-target[0]),'height':str(target[3]-target[1])})
        for gid in sorted(axis['data_ids'],key=lambda gid:drawing_order.get(gid,10**9)):
            element=all_ids.get(gid)
            if element is not None:
                destination=under if axis.get('data_roles',{}).get(gid) in ['QuadMesh','AxesImage'] else layer
                mark=transformed_group(element,source_rect,target,clip_id)
                if destination is under:mark=smooth_colorbar(mark,target,defs,f'repro_gradient_{gid}',clip_id)
                destination.append(mark)
        # Label slots store design coordinates and typography, never reference values.
        for j,label in enumerate(axis['labels']):
            text=re.sub(r'(?<![\d.])-0(?![\d.])','+0',label['text'])
            slot=design.get('labels',{}).get(label['gid'])
            lines=text.split('\n')
            if slot:
                for k,line in enumerate(lines):
                    part=slot[min(k,len(slot)-1)]
                    x=part['x'];y=part['y']
                    if 'point_anchor' in label:
                        px=target[0]+label['point_anchor'][0]*(target[2]-target[0]);py=target[1]+label['point_anchor'][1]*(target[3]-target[1])
                        x=px+part['dx'];y=py+part['dy']
                    face=part['face'];fill=part['fill']
                    if part.get('source_fontweight',label['fontweight'])!=label['fontweight']:
                        face=design.get('weight_faces',{}).get(label['fontweight'],face)
                    color=canonical_color(label['color'])
                    if part.get('source_color',color)!=color:fill=design.get('label_colors',{}).get(color,color)
                    add_text(layer,line,x,y,part['size'],face,fill)
            else:
                x=target[0]+label['anchor'][0]*(target[2]-target[0]);y=target[1]+label['anchor'][1]*(target[3]-target[1])
                face='bold' if label['fontweight']=='bold' else 'regular';size=design.get('label_size',label['fontsize'])
                for k,line in enumerate(lines):
                    width=sum((glyphs.get(face,glyphs['regular']).get(c) or glyphs['regular'].get(c) or {'advance':.3})['advance'] for c in line)*size
                    dx=-width/2 if label['ha']=='center' else -width if label['ha']=='right' else 0
                    add_text(layer,line,x+dx,y+size*.25+(k-(len(lines)-1)/2)*size, size,face,label['color'])
            label_audit.append({'id':label['gid'],'text':text,'uses_authored_slot':bool(slot)})
        if design.get('y_ticks'):
            positions=axis.get('ytick_anchors',[]);labels=axis['yticklabels']
            order=sorted(range(len(positions)),key=lambda j:positions[j])
            if len(order)==len(design['y_ticks'])+1 and labels.count('National')==1:
                order=[j for j in order if labels[j]!='National']
            if len(order)!=len(design['y_ticks']):
                raise ValueError(f'{key}: number of row labels differs from the manuscript layout')
            for j,slot in zip(order,design['y_ticks']):
                add_text(layer,labels[j],slot['x'],slot['y'],slot['size'],slot['face'],slot['fill'])
    if layout.get('m_tags'):
        layer=ET.SubElement(root,'{'+SVG+'}g',{'id':'repro_current_reference_counts'})
        labels=meta['axes'][0]['xticklabels']
        for label,slot in zip(labels,layout['m_tags']):
            add_text(layer,label.split('\n')[-1],slot['x'],slot['y'],slot['size'],slot['face'],slot['fill'])
    if layout.get('row_palette'):
        palette=json.loads((ROOT/'config/palette.json').read_text())
        axis=meta['axes'][0]
        order=sorted(range(len(axis['ytick_anchors'])),key=lambda j:axis['ytick_anchors'][j])
        if len(order)!=len(layout['row_palette']):raise ValueError('Row palette does not match the current matrix')
        layer=ET.SubElement(root,'{'+SVG+'}g',{'id':'repro_current_row_palette'})
        for j,rect in zip(order,layout['row_palette']):
            ET.SubElement(layer,'{'+SVG+'}rect',{'x':str(rect[0]),'y':str(rect[1]),
                'width':str(rect[2]-rect[0]),'height':str(rect[3]-rect[1]),'fill':palette[axis['yticklabels'][j]]})
    portable_svg(root)
    output_base=Path(output_base)
    live=deepcopy(root)
    # PDF typography is preserved using vector glyph outlines, no installed font required.
    outline_text(root,glyphs)
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    output_base.with_suffix('.svg').write_bytes(svg)
    output_base.with_name(output_base.name+'_text.svg').write_bytes(ET.tostring(live,encoding='utf-8',xml_declaration=True))
    import fitz
    import cairosvg
    pdf_bytes=cairosvg.svg2pdf(bytestring=svg)
    output_base.with_suffix('.pdf').write_bytes(pdf_bytes)
    pdf=fitz.open(stream=pdf_bytes,filetype='pdf')
    png=pdf[0].get_pixmap(matrix=fitz.Matrix(300/72,300/72),alpha=False).tobytes('png')
    output_base.with_suffix('.png').write_bytes(png)
    pdf.close()
    return {'figure':key,'source':'newly calculated table rendering','labels':label_audit,
            'reference_scientific_marks_used':False,'text_svg_requires_font':FAMILY,
            'font_license':'OFL-1.1','font_resource':'bundled unmodified Source Sans 3 TTF files'}
