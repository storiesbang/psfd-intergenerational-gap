import pandas as pd, numpy as np, statsmodels.api as sm, re
from statsmodels.stats.multitest import multipletests
exec(open('coef.py').read().split("G=[")[0])   # rebuild d with all factor columns
keep=['子女說心事（子報）','子女說心事（親報）','通話頻率','見面頻率',
      '金錢往來','家務協助天數','親代為父親']
s=d.copy()
s['金錢往來']=s['親→子 金錢']+s['子→親 金錢']
s['家務協助天數']=s['親→子 家務天數']+s['子→親 家務天數']
# structural missing: call/meet only exist for non-cohabitants (去年同住 skip). Code cohabitants as 0 after z, add dummy.
s['去年同住']=(s.call==0)*1.0; s.loc[s.call.isna(),'去年同住']=np.nan
X=pd.DataFrame(index=s.index)
for v in keep:
    x=s[v]; z=(x-x.mean())/x.std()
    if v in('通話頻率','見面頻率'): z=z.where(s['去年同住']==0,0.0)
    X[v]=z
X['同住']=(s['去年同住']-s['去年同住'].mean())/s['去年同住'].std()
ok=X.notna().all(axis=1)&s.gap.notna()
m=sm.OLS(s.gap[ok],sm.add_constant(X[ok])).fit(cov_type='cluster',cov_kwds={'groups':s.x01d[ok]})
ci=m.conf_int()
keep=keep[:4]+['同住']+keep[4:]
R=pd.DataFrame({'var':keep,'b':m.params[keep].values,'lo':ci.loc[keep,0].values,'hi':ci.loc[keep,1].values,'p':m.pvalues[keep].values})
R['q']=multipletests(R.p,method='fdr_bh')[1]
print('n',int(m.nobs),'R2 %.3f'%m.rsquared); open('n.txt','w').write(str(int(m.nobs)))
print(R.round(3).to_string()); R.to_csv('joint.csv',index=False)
