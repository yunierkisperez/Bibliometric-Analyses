"""Independent read-only comparison with user-supplied historical Excel files."""
from pathlib import Path
import numbers
import openpyxl

FLOATS={1:[3,4],2:[6,7],3:[8,9,10,11],4:[12,13,14,15],5:[12,13,14],
        6:[13,14,15,16,17,18,21],7:[4,5,6,9,11,12,13,14],8:[4],9:[5],10:[3],
        11:[4,8,10,14,17,18],12:[],13:[5,7,10,12,13,14,15,16],14:[4,7,10],15:[3]}


def missing(value):
    return value is None or value==''


def validate_tables(reference, candidate, tables=None):
    results=[]
    selected=list(range(1,16)) if tables is None else list(tables)
    if not selected or len(set(selected))!=len(selected) or any(n not in range(1,16) for n in selected):
        raise ValueError('Select distinct supplementary table numbers within 1-15')
    for n in selected:
        name=f'Table-S{n}.xlsx';errors=[];format_differences=0;cells=0;formulas=0
        paths=[Path(reference)/name,Path(candidate)/name]
        if not all(p.is_file() for p in paths):
            results.append({'table':n,'pass':False,'differences':[{'kind':'missing_workbook'}]});continue
        old,new=[openpyxl.load_workbook(p,data_only=False) for p in paths]
        oc,nc=[openpyxl.load_workbook(p,data_only=True) for p in paths]
        if old.sheetnames!=new.sheetnames:errors.append({'kind':'sheet_names_or_order'})
        for a,b,ac,bc in zip(old,new,oc,nc):
            if (a.max_row,a.max_column)!=(b.max_row,b.max_column):errors.append({'kind':'dimensions'})
            for ro,rn,vo,vn in zip(a,b,ac,bc):
                for x,y,v,w in zip(ro,rn,vo,vn):
                    cells+=1
                    if x.row==1 and x.value!=y.value:errors.append({'kind':'header','cell':x.coordinate})
                    if x.data_type=='f':
                        formulas+=1
                        if x.value!=y.value:errors.append({'kind':'formula','cell':x.coordinate})
                    elif y.data_type=='f':errors.append({'kind':'unexpected_formula','cell':x.coordinate})
                    v=v.value;w=w.value
                    if missing(v) and missing(w):equal=True
                    elif isinstance(v,numbers.Real) and not isinstance(v,bool) and isinstance(w,numbers.Real) and not isinstance(w,bool):
                        tol=1e-12+5e-14*abs(v) if x.column-1 in FLOATS[n] else 0
                        equal=abs(v-w)<=tol
                    else:equal=type(v)==type(w) and v==w
                    if not equal:errors.append({'kind':'scientific_value_or_type','cell':x.coordinate})
                    if x.number_format!=y.number_format:format_differences+=1
        for w in [old,new,oc,nc]:w.close()
        results.append({'table':n,'pass':not errors and format_differences==0,'cells_checked':cells,'formula_cells':formulas,
                        'difference_count':len(errors),'differences':errors[:40],
                        'number_format_differences':format_differences,
                        'float_tolerance':'1e-12 + 5e-14*abs(reference), enumerated float columns only; counts and IDs exact'})
    return results
