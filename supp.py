import pandas as pd, numpy as np, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
# reproduce joint_7 exactly (no file writes)
src=open('joint_7.py').read().replace("open('n.txt','w').write(str(int(m.nobs)))","None").replace("R.to_csv('joint.csv',index=False)","None")
exec(src)
base_s, base_X, base_ok, base_m, base_R = s.copy(), X.copy(), ok.copy(), m, R.copy()
pd.set_option('display.width',220)
print('\n### 1. Dependent variable')
y=s.gap[ok]
print('y = s.gap (joint_7 L18), gap = p_rating - c_rating (match.py L34); not standardized')
print('n',len(y),'min',y.min(),'max',y.max(),'mean %.4f'%y.mean(),'sd %.4f'%y.std())
print('value counts',y.value_counts().sort_index().to_dict())
print('\n### 2. housework exclusion')
pre=ok.copy(); pre[:]=X[['子女說心事（子報）','通話頻率','見面頻率','金錢往來','親代為父親','同住']].notna().all(axis=1)
exc=s[pre & X['家務協助天數'].isna()]
print('excluded for housework (given others present):',len(exc))
def rawslot(var,rows): return [raw.at[r.p_id,var.format(k=int(r.slot))] for _,r in rows.iterrows()]
e=pd.DataFrame({'g24':rawslot('g24c{k}',exc),'g25':rawslot('g25c{k}',exc)})
print('g24 raw',e.g24.value_counts(dropna=False).sort_index().to_dict()); print('g25 raw',e.g25.value_counts(dropna=False).sort_index().to_dict())
print(pd.crosstab(e.g24.astype(str),e.g25.astype(str)))
allv=pd.Series(dtype=float)
for k in range(1,7):
    for v in ['g24c','g25c']: allv=pd.concat([allv,pd.to_numeric(raw[f'{v}{k}'],errors='coerce')])
print('ALL g24/g25 values in raw file >30 or <0:',allv[(allv>30)|(allv<0)].value_counts().sort_index().to_dict(), '| max',allv.max())
print('\n### 3. cohab definitions (958)')
S=s[ok]; h13=(S.call==0); h11=(S.c_dist==0)  # c_dist: h11{s}1 with 8 recoded to 0 (ext.py L19,L24)
print(pd.crosstab(h13.rename('h13==0'),h11.where(S.c_dist.notna()).rename('h11==8'),dropna=False,margins=True))
def fit(X,s,keep):
    ok=X.notna().all(axis=1)&s.gap.notna()
    m=sm.OLS(s.gap[ok],sm.add_constant(X[ok])).fit(cov_type='cluster',cov_kwds={'groups':s.x01d[ok]})
    ci=m.conf_int(); R=pd.DataFrame({'var':keep,'b':m.params[keep].values,'lo':ci.loc[keep,0].values,'hi':ci.loc[keep,1].values,'p':m.pvalues[keep].values})
    R['q']=multipletests(R.p,method='fdr_bh')[1]; return int(m.nobs),ok,R
X3=base_X.copy(); hh=np.where(s.c_dist.isna(),np.nan,(s.c_dist==0)*1.0); hh=pd.Series(hh,index=s.index)
X3['同住']=(hh-hh.mean())/hh.std()
n3,ok3,R3=fit(X3,s,list(base_R['var']))
print('h11 version n',n3); print(R3.round(4).to_string())
print('\n### 4. split money & housework (9 coefs)')
X4=base_X.drop(columns=['金錢往來','家務協助天數']).copy()
for v in ['親→子 金錢','子→親 金錢','親→子 家務天數','子→親 家務天數']:
    x=s[v]; X4[v]=(x-x.mean())/x.std()
keep4=['子女說心事（子報）','通話頻率','見面頻率','同住','親→子 金錢','子→親 金錢','親→子 家務天數','子→親 家務天數','親代為父親']
n4,ok4,R4=fit(X4[keep4],s,keep4); print('n',n4); print(R4.round(4).to_string())
print('\n### 6. FDR & clusters')
print('method fdr_bh via statsmodels multipletests (joint_7 L22); clusters (unique x01d) in 958:',s.x01d[ok].nunique())
print('baseline disc', base_R.iloc[0].to_dict())
