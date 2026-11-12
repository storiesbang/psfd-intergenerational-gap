import pandas as pd, numpy as np, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
U="/root/.claude/uploads/77e8561f-a7bf-5fe7-a84b-17de4d9fbf63/"
raw=pd.read_csv(U+"d7cf9816-1791520728709_psfd_rr2024_v202603_csv.csv",low_memory=False).set_index('x01')
B=pd.read_csv('dyad_ext.csv'); d=B[B.wave==2024].dropna(subset=['gap']).copy()
def num(v):
    try: v=float(v)
    except: return np.nan
    return v
def slot(var,lo,hi): return [ (lambda v: v if lo<=v<=hi else np.nan)(num(raw.at[r.p_id,var.format(k=int(r.slot))])) for _,r in d.iterrows()]
def par(var,lo,hi): return [ (lambda v: v if lo<=v<=hi else np.nan)(num(raw.at[r.p_id,var])) for _,r in d.iterrows()]
def kid(var,lo,hi): return [ (lambda v: v if lo<=v<=hi else np.nan)(num(raw.at[r.c_id,var.format(s='f' if r.p_sex==1 else 'm')])) for _,r in d.iterrows()]
edu=np.array(slot('g16c{k}',1,16))
d['子代大學以上']=np.where(np.isnan(edu),np.nan,((edu>=12)&(edu<=15))*1.0)
m=np.array(slot('g29c{k}',1,5)); d['子代已婚']=np.where(np.isnan(m),np.nan,(m==2)*1.0)
j=np.array(slot('g21c{k}',1,3)); d['子代有工作']=np.where(np.isnan(j),np.nan,(j<=2)*1.0)
d['親→子 家務天數']=slot('g24c{k}',0,30); d['子→親 家務天數']=slot('g25c{k}',0,30)
x=np.array(slot('g22z01c{k}' if 'g22z01c1' in raw.columns else 'g22c{k}',1,2)); d['親→子 金錢']=np.where(np.isnan(x),np.nan,(x==1)*1.0)
x=np.array(slot('g23z01c{k}',1,2)); d['子→親 金錢']=np.where(np.isnan(x),np.nan,(x==1)*1.0)
h=np.array(par('a11',1,5)); d['親代健康']=6-h
d['子女數']=par('g02z01',1,15)
d['親代為父親']=(d.p_sex==1)*1.0; d['子代為兒子']=(d.c_sex==1)*1.0
d['親代年齡']=d.p_age
d['子女說心事（子報）']=6-d.c_disc_cp; d['子女說心事（親報）']=6-d.p_disc_cp
d['父母說心事（子報）']=6-d.c_disc_pc; d['父母說心事（親報）']=6-d.p_disc_pc
d['通話頻率']=np.where(d.call>=1,6-d.call,np.nan)   # non-cohab only
d['見面頻率']=7-d.meet
d['同住']=np.where(d.c_dist.isna(),np.nan,(d.c_dist==0)*1.0)
G=[('互動品質',['子女說心事（子報）','子女說心事（親報）','父母說心事（子報）','父母說心事（親報）']),
   ('接觸量',['通話頻率','見面頻率','同住']),
   ('資源交換',['親→子 金錢','子→親 金錢','親→子 家務天數','子→親 家務天數']),
   ('人口與社經',['親代為父親','子代為兒子','親代年齡','親代健康','子女數','子代大學以上','子代已婚','子代有工作'])]
rows=[]
for grp,vs in G:
    for v in vs:
        s=d.dropna(subset=[v,'gap']).copy(); s['z']=(s[v]-s[v].mean())/s[v].std()
        m1=smf.ols('gap~z',s).fit(cov_type='cluster',cov_kwds={'groups':s.x01d})
        row=dict(group=grp,var=v,n=int(m1.nobs),b=m1.params.z,lo=m1.conf_int().loc['z',0],hi=m1.conf_int().loc['z',1],p=m1.pvalues.z)
        if v!='子女說心事（子報）':
            t=s.dropna(subset=['子女說心事（子報）']).copy(); t['z']=(t[v]-s[v].mean())/s[v].std(); t['dz']=(t['子女說心事（子報）']-d['子女說心事（子報）'].mean())/d['子女說心事（子報）'].std()
            m2=smf.ols('gap~z+dz',t).fit(cov_type='cluster',cov_kwds={'groups':t.x01d})
            row.update(b2=m2.params.z,lo2=m2.conf_int().loc['z',0],hi2=m2.conf_int().loc['z',1],p2=m2.pvalues.z,n2=int(m2.nobs))
        rows.append(row)
R=pd.DataFrame(rows); R['q']=multipletests(R.p,method='fdr_bh')[1]
m=R.p2.notna(); R.loc[m,'q2']=multipletests(R.loc[m,'p2'],method='fdr_bh')[1]
R.to_csv('coef.csv',index=False)
pd.set_option('display.width',200); print(R.round(3).to_string())
