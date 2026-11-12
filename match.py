import pandas as pd, numpy as np, sys
U="/root/.claude/uploads/77e8561f-a7bf-5fe7-a84b-17de4d9fbf63/"
檔案={2020:U+"7737d9a9-1791520839885_psfd_rr2020_v202208_csv.csv",2022:U+"6f51fc00-1791520817517_psfd_rr2022_v202403_csv.csv",2024:U+"d7cf9816-1791520728709_psfd_rr2024_v202603_csv.csv"}
新式={'slot_sex':'g07c{k}','slot_by':'g08z01c{k}','slot_bm':'g08z02c{k}','p_rating':'g28c{k}','p_disc_pc':'g26c{k}','p_disc_cp':'g27c{k}','c_rating':'h19{s}1','c_disc_pc':'h17{s}1','c_disc_cp':'h18{s}1','call':'h13{s}1','meet':'h14{s}1','r_by':'a02z01','r_bm':'a02z02','happy':'a06'}
舊式={'slot_sex':'b12bc{k}','slot_by':'b12c01z01c{k}','slot_bm':'b12c01z02c{k}','p_rating':'b12m04c{k}','p_disc_pc':'b12m03c{k}','p_disc_cp':'b12n03c{k}','c_rating':'b16h{s}1','c_disc_pc':'b16e{s}1','c_disc_cp':'b16f{s}1','call':'b16a{s}1','meet':'b16b{s}1','r_by':'a02a','r_bm':'a02b','happy':'w01a'}
對照={2020:舊式,2022:新式,2024:新式}
有效範圍_2020={'slot_sex':(1,2),'slot_by':(20,115),'slot_bm':(1,12),'p_rating':(1,5),'p_disc_pc':(1,5),'p_disc_cp':(1,5),'c_rating':(1,5),'c_disc_pc':(1,5),'c_disc_cp':(1,5),'call':(0,5),'meet':(1,6),'r_by':(20,115),'r_bm':(1,12),'happy':(1,7)}
def 清(欄,名稱,wave):
    if wave==2020 and 名稱 in 有效範圍_2020:
        lo,hi=有效範圍_2020[名稱]; return 欄.where((欄>=lo)&(欄<=hi))
    return 欄.where(欄>=0)
def 配對(wave):
    M=對照[wave]; df=pd.read_csv(檔案[wave],low_memory=False); df.columns=[c.lstrip('﻿') for c in df.columns]
    父母=df[df.x01a==1]; 子女=df[df.x01a==3]; 片段=[]
    for k in range(1,7):
        t=pd.DataFrame({'p_id':父母.x01.values,'x01d':父母.x01d.values,'p_sex':父母.a01.values,'slot':k})
        for n in ['slot_sex','slot_by','slot_bm','p_rating','p_disc_pc','p_disc_cp']:
            t[n]=清(父母[M[n].format(k=k)],n,wave).values
        片段.append(t)
    P=pd.concat(片段,ignore_index=True); P=P[P.slot_by.notna()|P.p_rating.notna()]
    子=pd.DataFrame({'c_id':子女.x01.values,'x01d':子女.x01d.values,'c_seq':子女.x01e.values,'c_sex':子女.a01.values})
    子['c_by']=清(子女[M['r_by']],'r_by',wave).values; 子['c_bm']=清(子女[M['r_bm']],'r_bm',wave).values; 子['happy']=清(子女[M['happy']],'happy',wave).values
    for n in ['c_rating','c_disc_pc','c_disc_cp','call','meet']:
        for s in 'fm': 子[f'{n}_{s}']=清(子女[M[n].format(s=s)],n,wave).values
    C=P.merge(子,on='x01d'); C=C[(C.slot_sex==C.c_sex)&(C.slot_by==C.c_by)].copy()
    C['月合']=(C.slot_bm==C.c_bm).astype(int)
    C=C.sort_values(['x01d','月合','slot','c_seq'],ascending=[True,False,True,True],kind='mergesort')
    a,b,keep=set(),set(),[]
    for i,r in C.iterrows():
        if (r.p_id,r.slot) in a or r.c_id in b: continue
        a.add((r.p_id,r.slot)); b.add(r.c_id); keep.append(i)
    d=C.loc[keep].copy(); f=d.p_sex==1
    for n in ['c_rating','c_disc_pc','c_disc_cp','call','meet']: d[n]=np.where(f,d[f'{n}_f'],d[f'{n}_m'])
    d['wave']=wave; d['dyad_id']=d.x01d.astype(int).astype(str)+'_'+d.c_seq.astype(int).astype(str); d['gap']=d.p_rating-d.c_rating
    return d[['dyad_id','wave','x01d','c_seq','p_id','c_id','slot','p_sex','c_sex','c_by','p_rating','c_rating','gap','p_disc_pc','p_disc_cp','c_disc_pc','c_disc_cp','call','meet','happy']]
預期={2020:932,2022:1000,2024:997}; L=[]
for w in [2020,2022,2024]:
    d=配對(w); d.to_csv(f'dyad{w}.csv',index=False,encoding='utf-8-sig'); L.append(d); h=d.dropna(subset=['p_rating','c_rating'])
    print(w,len(d),預期[w],'戶',d.x01d.nunique(),'有分',len(h),f'gap {h.gap.mean():+.3f} r {h.p_rating.corr(h.c_rating):.3f}')
A=pd.concat(L,ignore_index=True); A.to_csv('dyad_all.csv',index=False,encoding='utf-8-sig')
n=A.groupby('dyad_id').wave.nunique(); print('3波',(n==3).sum(),'戶',A[A.dyad_id.isin(n[n==3].index)].x01d.nunique())
for w in [2020,2022,2024]: print(w,'call codes',A[A.wave==w].call.value_counts(dropna=False).sort_index().to_dict())
