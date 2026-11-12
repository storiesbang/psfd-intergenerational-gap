import pandas as pd, numpy as np
U="/root/.claude/uploads/77e8561f-a7bf-5fe7-a84b-17de4d9fbf63/"
F={2020:U+"7737d9a9-1791520839885_psfd_rr2020_v202208_csv.csv",2022:U+"6f51fc00-1791520817517_psfd_rr2022_v202403_csv.csv",2024:U+"d7cf9816-1791520728709_psfd_rr2024_v202603_csv.csv"}
A=pd.read_csv('dyad_all.csv')
out=[]
for w in [2020,2022,2024]:
    df=pd.read_csv(F[w],low_memory=False); df.columns=[c.lstrip('﻿') for c in df.columns]
    d=A[A.wave==w].copy()
    pc = 'b12p01c{k}' if w==2020 else 'g32c{k}'
    cc = 'b15b{s}1' if w==2020 else 'h11{s}1'
    by = 'a02a' if w==2020 else 'a02z01'
    idx=df.set_index('x01')
    def val(v,lo,hi):
        return v if (v is not None and lo<=v<=hi) else np.nan
    pd_,cd_,page=[],[],[]
    for _,r in d.iterrows():
        pd_.append(val(idx.at[r.p_id,pc.format(k=int(r.slot))],1,8))
        s='f' if r.p_sex==1 else 'm'
        cd_.append(val(idx.at[r.c_id,cc.format(s=s)],1,8))
        page.append(w-1911-idx.at[r.p_id,by] if 20<=idx.at[r.p_id,by]<=115 else np.nan)
    d['p_dist']=pd_; d['c_dist']=cd_; d['p_age']=page
    out.append(d)
B=pd.concat(out)
for c in ['p_dist','c_dist']: B[c]=B[c].replace(8,0)  # 同住=0
B['c_age']=B.wave-1911-B.c_by
B.to_csv('dyad_ext.csv',index=False)
print(B.groupby('wave')[['p_dist','c_dist','p_age','c_age']].describe().T.round(2).iloc[:,:])
