import pandas as pd, numpy as np, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
# build d exactly as joint_7 does (coef.py lines before "G=[")
exec(open('coef.py').read().split("G=[")[0])
s=d.copy()
s['金錢往來']=s['親→子 金錢']+s['子→親 金錢']; s['家務協助天數']=s['親→子 家務天數']+s['子→親 家務天數']
s['去年同住']=(s.call==0)*1.0; s.loc[s.call.isna(),'去年同住']=np.nan
OTH=['通話頻率','見面頻率','金錢往來','家務協助天數','親代為父親']
def run(disc, y, sub=None):
    X=pd.DataFrame(index=s.index)
    for v in disc+OTH:
        x=s[v]; z=(x-x.mean())/x.std()
        if v in('通話頻率','見面頻率'): z=z.where(s['去年同住']==0,0.0)
        X[v]=z
    X['同住']=(s['去年同住']-s['去年同住'].mean())/s['去年同住'].std()
    ok=X.notna().all(axis=1)&s[y].notna()
    if sub is not None: ok&=sub
    m=sm.OLS(s[y][ok],sm.add_constant(X[ok])).fit(cov_type='cluster',cov_kwds={'groups':s.x01d[ok]})
    keep=disc+OTH[:2]+['同住']+OTH[2:]
    p=m.pvalues[keep]; q=pd.Series(multipletests(p,method='fdr_bh')[1],index=keep)
    ci=m.conf_int()
    return {v:(m.params[v],ci.loc[v,0],ci.loc[v,1],p[v],q[v]) for v in disc}, int(m.nobs), s.x01d[ok].nunique()
C,P='子女說心事（子報）','子女說心事（親報）'
rows=[]
for label,disc,sub in [('A 子報版',[C],None),('B 親報版',[P],None),('C 兩者同放',[C,P],None),
                        ('D 一致子集・子報版',[C],(s.c_disc_cp==s.p_disc_cp))]:
    for y,yl in [('gap','溫差'),('p_rating','親評'),('c_rating','子評')]:
        res,n,ncl=run(disc,y,sub)
        for v,(b,lo,hi,p,q) in res.items():
            rows.append(dict(模型=label,依變項=yl,係數=v.replace('子女說心事',''),n=n,clusters=ncl,b=b,lo=lo,hi=hi,p=p,q=q))
R=pd.DataFrame(rows); pd.set_option('display.width',250)
print(R.to_string(formatters={'b':'{:+.3f}'.format,'lo':'{:+.3f}'.format,'hi':'{:+.3f}'.format,'p':'{:.3g}'.format,'q':'{:.3g}'.format}))
both=s.dropna(subset=[C,P]); print('r(子報,親報) among d:',round(both[C].corr(both[P]),3),'n',len(both))
X0=s.dropna(subset=['gap',C,P]); print('concordant share', round((X0.c_disc_cp==X0.p_disc_cp).mean(),3), 'n concordant', int((X0.c_disc_cp==X0.p_disc_cp).sum()))
R.to_csv('supp2_results.csv',index=False)
