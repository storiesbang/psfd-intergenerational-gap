# 驗 supp2 模型 C：親評←親報 0.174 vs 子評←子報 0.262 的差（堆疊兩式、戶層叢集）
import io, contextlib, numpy as np, pandas as pd, statsmodels.api as sm
import os
src=open("psfd_controls_robust.py",encoding="utf-8").read().split("# ── 3.")[0]
ns={"__file__":os.path.abspath("psfd_controls_robust.py")}; exec(src,ns)
s=ns["d"].copy()
s['金錢往來']=s['親→子 金錢']+s['子→親 金錢']; s['家務協助天數']=s['親→子 家務天數']+s['子→親 家務天數']
s['去年同住']=(s.call==0)*1.0; s.loc[s.call.isna(),'去年同住']=np.nan
C,P='子女說心事（子報）','子女說心事（親報）'
OTH=['通話頻率','見面頻率','金錢往來','家務協助天數','親代為父親']
X=pd.DataFrame(index=s.index)
for v in [C,P]+OTH:
    x=s[v]; z=(x-x.mean())/x.std()
    if v in('通話頻率','見面頻率'): z=z.where(s['去年同住']==0,0.0)
    X[v]=z
X['同住']=(s['去年同住']-s['去年同住'].mean())/s['去年同住'].std()
ok=X.notna().all(axis=1)&s.p_rating.notna()&s.c_rating.notna()
print("口徑：supp2 模型 C 設計｜n=",int(ok.sum()),"｜SD 子報",round(s[C].std(),3),"親報",round(s[P].std(),3))
# 堆疊：每對兩列（親評、子評），所有係數依方程式交乘
Xo=sm.add_constant(X[ok]); n=len(Xo)
Zp=np.vstack([np.hstack([Xo.values,np.zeros_like(Xo.values)]),np.hstack([np.zeros_like(Xo.values),Xo.values])])
cols=[f"p:{c}" for c in Xo.columns]+[f"c:{c}" for c in Xo.columns]
y=np.concatenate([s.p_rating[ok].values,s.c_rating[ok].values]); g=np.concatenate([s.x01d[ok].values]*2)
m=sm.OLS(y,pd.DataFrame(Zp,columns=cols)).fit(cov_type='cluster',cov_kwds={'groups':g})
bp,bc=m.params[f"p:{P}"],m.params[f"c:{C}"]
t=m.t_test(np.eye(len(cols))[cols.index(f"p:{P}")]-np.eye(len(cols))[cols.index(f"c:{C}")])
print(f"親評←親報 {bp:+.3f}｜子評←子報 {bc:+.3f}｜差 {bp-bc:+.3f} p={float(t.pvalue):.3f}｜比 {bp/bc:.2f}")