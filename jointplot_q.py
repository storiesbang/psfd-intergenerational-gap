import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.transforms as mt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']='Noto Sans CJK JP'; plt.rcParams['axes.unicode_minus']=False
R=pd.read_csv('joint.csv'); n=int(open('n.txt').read())
SIG,NS,MUTED='#184f95','#9a9fa5','#777'
fig,ax=plt.subplots(figsize=(8.2,4.2),dpi=200)
T=mt.blended_transform_factory(ax.transAxes,ax.transData)
for i,r in R.iterrows():
    c=SIG if r.q<.05 else NS
    ax.plot([r.lo,r.hi],[i,i],color=c,lw=2.2,zorder=2)
    ax.scatter(r.b,i,s=60,zorder=3,facecolor=c if r.q<.05 else 'white',edgecolor=c,linewidth=1.6)
    ax.text(1.07,i,f'{r.b:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]',va='center',fontsize=10.4,color='#222' if r.q<.05 else '#555',family='DejaVu Sans Mono',transform=T)
    ax.text(2.22,i,('9.8×10⁻⁸' if r.q<0.001 else f'{r.q:.3f}'),va='center',fontsize=10.4,color='#222' if r.q<.05 else '#555',family='DejaVu Sans Mono',transform=T)
ax.text(1.07,-1.05,'係數 [95% CI]',fontsize=11.7,fontweight='bold',transform=T); ax.text(2.22,-1.05,'q 值',fontsize=11.7,fontweight='bold',transform=T)
ax.axvline(0,color='#555',lw=.9)
ax.set_yticks(range(len(R))); ax.set_yticklabels([v+('＊' if v in('通話頻率','見面頻率') else '') for v in R['var']],fontsize=12.3)
ax.tick_params(axis='y',length=0); ax.set_ylim(len(R)-0.4,-1.4); ax.set_xlim(-0.30,0.30)
ax.set_xlabel('因素每增加 1 個標準差，溫差的變化（級）\n← 溫差縮小　　　　　　溫差擴大 →',fontsize=12.3)
ax.grid(axis='x',color='#e6e6e6',lw=.6); ax.set_axisbelow(True)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#888'); ax.tick_params(axis='x',labelsize=9,colors='#444')
fig.text(0.02,0.02,f'註：RR2024，n = {n} 對，所有因素同時放入模型並標準化；誤差線為 95% CI。\nq 值為 BH–FDR 校正後之 p 值；實心＝q < 0.05，空心＝不顯著。＊僅非同住者有值。',fontsize=9.8,color=MUTED,linespacing=1.5)
fig.subplots_adjust(left=0.21,right=0.5,top=0.93,bottom=0.31)
fig.savefig('fig_joint_q.png',dpi=200,facecolor='white')
