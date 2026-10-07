from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata

SEED=104729
COMPLETE=[25,50,75,100,110,120]
LAYERS=['layer_1','layer_2','layer_3','layer_4','layer_5','layer_6','final']
LORD={x:i+1 for i,x in enumerate(LAYERS)}
DATA=Path('/mnt/data/followup_work/dataset.json')
OUT=Path('/mnt/data/followup_work/results'); OUT.mkdir(parents=True,exist_ok=True)
D=json.loads(DATA.read_text())

def ci(a):
    a=np.asarray(a,float); return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]
def corr(a,b,kind='spearman'):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3 or np.std(a[m])<1e-12 or np.std(b[m])<1e-12:return float('nan')
    return float(spearmanr(a[m],b[m]).statistic if kind=='spearman' else pearsonr(a[m],b[m]).statistic)
def ols(X,y):
    X=np.asarray(X,float);y=np.asarray(y,float); b=np.linalg.lstsq(X,y,rcond=None)[0];p=X@b;r=y-p
    sse=float(r@r);sst=float(((y-y.mean())**2).sum());r2=1-sse/sst if sst else np.nan
    return b,p,r,sse,r2
def checkpoint(v,t):return f'{v}@{t}M'

def cluster_boot_mean(df,col,B,seed):
    g=df.groupby('checkpoint')[col].agg(['mean','count'])
    means=g['mean'].to_numpy(float); counts=g['count'].to_numpy(float)
    # all current uses have equal counts, but retain weighted form
    rng=np.random.default_rng(seed); G=len(g)
    C=rng.multinomial(G,np.full(G,1/G),size=B)
    num=C@(means*counts);den=C@counts
    return ci(num/den)

# flatten primitives
rows=[]
for rep in D['full_layer_reports'].values():
    v=rep['variant_id'];t=int(rep['tokens_m'])
    for ln in LAYERS:
        m=rep['layers'][ln]
        r={'variant':v,'tokens_m':t,'checkpoint':checkpoint(v,t),'layer':ln,'layer_ord':LORD[ln],
           'B_condition_1_accuracy':m['B_condition_1_accuracy'],'B_under_C_accuracy':m['B_under_C_accuracy'],'B_under_D_accuracy':m['B_under_D_accuracy'],
           'B_cluster_margin':m['B_cluster']['margin'],'B_cluster_within_cosine':m['B_cluster']['within_cosine'],'B_cluster_between_cosine':m['B_cluster']['between_cosine'],
           'C_relation_margin':m['C_relative_to_B']['relation_margin'],'C_same_offset_cosine':m['C_relative_to_B']['same_goal_offset_cosine'],'C_diff_centroid_cosine':m['C_relative_to_B']['different_goal_centroid_cosine'],
           'A_axis_consistency':m['A_axis_consistency'],'A_effect_magnitude':m['A_effect_magnitude']}
        for mode in ['zero','negate','shuffle']:
            r[f'B_A_{mode}_accuracy']=m['B_intervention_accuracy_by_A_mode'][mode]
            r[f'A_{mode}_axis_consistency']=m['A_intervention_geometry'][mode]['axis_consistency']
            r[f'A_{mode}_effect_norm']=m['A_intervention_geometry'][mode]['mean_effect_norm']
        lr=m.get('learned_A_reference')
        for ref in ['learned','shuffle','random']:
            r[f'B_ref_{ref}_margin']=np.nan if lr is None else lr['relative_B_margin_by_reference'][ref]
            r[f'C_ref_{ref}_margin']=np.nan if lr is None else lr['relative_C_relation_margin_by_reference'][ref]
        rows.append(r)
L=pd.DataFrame(rows);L.to_csv(OUT/'all_primitive_layer_rows.csv',index=False)
summary={'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'seed':SEED,'complete_tokens':COMPLETE,'benchmarks':{}}

# F1
base=L[L.tokens_m.isin(COMPLETE)&(L.variant!='V1')]
f=[]
for _,r in base.iterrows():
    g=f"{r.checkpoint}|{r.layer}"
    for mode in ['zero','negate','shuffle']:
        f.append({'variant':r.variant,'tokens_m':r.tokens_m,'checkpoint':r.checkpoint,'layer':r.layer,'group':g,'mode':mode,
                  'B_accuracy':r[f'B_A_{mode}_accuracy'],'effect_norm':r[f'A_{mode}_effect_norm'],'axis_consistency':r[f'A_{mode}_axis_consistency']})
F1=pd.DataFrame(f)
for c in ['B_accuracy','effect_norm']:
    F1[c+'_c']=F1[c]-F1.groupby('group')[c].transform('mean')
F1['negate']=(F1['mode']=='negate').astype(float);F1['shuffle']=(F1['mode']=='shuffle').astype(float)
for c in ['negate','shuffle']:F1[c+'_c']=F1[c]-F1.groupby('group')[c].transform('mean')
X1=F1[['effect_norm_c']].to_numpy();X3=F1[['effect_norm_c','negate_c','shuffle_c']].to_numpy();y=F1.B_accuracy_c.to_numpy()
b1,p1,r1,sse1,r21=ols(X1,y);b3,p3,r3,sse3,r23=ols(X3,y)
F1['pred_mag_only_c']=p1;F1['pred_mag_mode_c']=p3;F1['resid_mag_only']=r1;F1['resid_mag_mode']=r3
# group sufficient stats, 3 rows each
Gs=list(F1.group.unique());G=len(Gs)
XtX=[];Xty=[];XtX1=[];Xty1=[]; idxs=[]
for g in Gs:
    ix=np.flatnonzero(F1.group.to_numpy()==g);idxs.append(ix)
    X=X3[ix];xx=X1[ix];yy=y[ix]
    XtX.append(X.T@X);Xty.append(X.T@yy);XtX1.append(float(xx.T@xx));Xty1.append(float(xx.T@yy))
XtX=np.asarray(XtX);Xty=np.asarray(Xty);XtX1=np.asarray(XtX1);Xty1=np.asarray(Xty1)
TX=XtX.sum(0);Ty=Xty.sum(0);TX1=XtX1.sum();Ty1=Xty1.sum()
# leave-one-group-out using sufficient stats
cv=[]
for gi,g in enumerate(Gs):
    bf=np.linalg.solve(TX-XtX[gi],Ty-Xty[gi]); bm=(Ty1-Xty1[gi])/(TX1-XtX1[gi])
    ix=idxs[gi];pf=X3[ix]@bf;pm=X1[ix,0]*bm
    for j,k in enumerate(ix):cv.append({'index':int(k),'group':g,'sqerr_mag':float((y[k]-pm[j])**2),'sqerr_full':float((y[k]-pf[j])**2)})
CV=pd.DataFrame(cv);cvs1=float(CV.sqerr_mag.sum());cvs3=float(CV.sqerr_full.sum())
# multinomial cluster bootstrap sufficient stats
rng=np.random.default_rng(SEED+1);C=rng.multinomial(G,np.full(G,1/G),size=2000)
BTX=np.einsum('bg,gij->bij',C,XtX);BTy=np.einsum('bg,gj->bj',C,Xty)
boot=np.linalg.solve(BTX,BTy[...,None])[...,0]
F1.to_csv(OUT/'F1_direction_vs_magnitude_raw.csv',index=False);CV.to_csv(OUT/'F1_leave_group_out_raw.csv',index=False)
summary['benchmarks']['F1']={'n_checkpoint_layers':G,'n_rows':len(F1),'magnitude_only_beta':float(b1[0]),'magnitude_only_r2_within':r21,'magnitude_only_sse':sse1,
 'full_beta_effect_norm':float(b3[0]),'full_beta_negate':float(b3[1]),'full_beta_shuffle':float(b3[2]),
 'full_beta_ci95':{'effect_norm':ci(boot[:,0]),'negate':ci(boot[:,1]),'shuffle':ci(boot[:,2])},'full_r2_within':r23,'full_sse':sse3,
 'sse_improvement_fraction':float((sse1-sse3)/sse1),'leave_group_out_sse_mag_only':cvs1,'leave_group_out_sse_mag_plus_mode':cvs3,'leave_group_out_improvement_fraction':float((cvs1-cvs3)/cvs1)}

# F2
F2=L[L.tokens_m.isin(COMPLETE)&(L.variant!='V1')].copy()
for target in ['B','C']:
    F2[f'{target}_learned_vs_mean_alt']=F2[f'{target}_ref_learned_margin']-(F2[f'{target}_ref_shuffle_margin']+F2[f'{target}_ref_random_margin'])/2
    F2[f'{target}_learned_vs_shuffle']=F2[f'{target}_ref_learned_margin']-F2[f'{target}_ref_shuffle_margin']
    F2[f'{target}_learned_vs_random']=F2[f'{target}_ref_learned_margin']-F2[f'{target}_ref_random_margin']
    F2[f'{target}_shuffle_vs_random']=F2[f'{target}_ref_shuffle_margin']-F2[f'{target}_ref_random_margin']
F2.to_csv(OUT/'F2_reference_specificity_raw.csv',index=False)
f2={}
for target in ['B','C']:
    col=f'{target}_learned_vs_mean_alt';a=F2[col].dropna().to_numpy(float)
    pv={}
    for v,z in F2.groupby('variant'):
        q=z[col].dropna().to_numpy(float);pv[v]={'mean':float(q.mean()),'median':float(np.median(q)),'positive_fraction':float((q>0).mean()),'negative_fraction':float((q<0).mean()),'n':len(q)}
    f2[target]={'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'ci95_cluster_checkpoint':cluster_boot_mean(F2,col,3000,SEED+2),
                'positive_fraction':float((a>0).mean()),'negative_fraction':float((a<0).mean()),'shuffle_random_noise_mean_abs':float(F2[f'{target}_shuffle_vs_random'].abs().mean()),
                'learned_adv_mean_abs':float(F2[col].abs().mean()),'per_variant':pv}
summary['benchmarks']['F2']=f2

# F3
m=[]
for rep in D['full_layer_reports'].values():
    v=rep['variant_id'];t=int(rep['tokens_m']);diag=rep.get('A_mechanism_diagnostics')
    if v=='V1' or t not in COMPLETE or not diag:continue
    for i,p in enumerate(diag['layer_parameters']):
        ln=f'layer_{i+1}';lm=rep['layers'][ln]
        m.append({'variant':v,'tokens_m':t,'checkpoint':checkpoint(v,t),'layer':ln,'layer_ord':i+1,'A_vector_norm':diag['A_vector_norm'],'gate':p['gate'],'path_up_norm':p['path_up_norm'],'path_down_norm':p['path_down_norm'],'A_effect_magnitude':lm['A_effect_magnitude']})
F3=pd.DataFrame(m)
for c in ['gate','path_up_norm','path_down_norm','A_effect_magnitude']:F3[c+'_c']=F3[c]-F3.groupby('checkpoint')[c].transform('mean')
F3.to_csv(OUT/'F3_mechanism_coupling_raw.csv',index=False)
cor={c:{'pearson_centered':corr(F3[c+'_c'],F3.A_effect_magnitude_c,'pearson'),'spearman_centered':corr(F3[c+'_c'],F3.A_effect_magnitude_c,'spearman')} for c in ['gate','path_up_norm','path_down_norm']}
def standard(a):
    a=np.asarray(a,float);s=a.std(0);s=np.where(s<1e-12,1,s);return (a-a.mean(0))/s
Xs=standard(F3[['gate_c','path_up_norm_c','path_down_norm_c']]);ys=standard(F3[['A_effect_magnitude_c']])[:,0]
bs,_,_,_,r2s=ols(Xs,ys)
lag=[]
for cp,z in F3.sort_values('layer_ord').groupby('checkpoint'):
    z=z.sort_values('layer_ord').reset_index(drop=True)
    for i in range(5):lag.append({'checkpoint':cp,'variant':z.loc[i,'variant'],'tokens_m':z.loc[i,'tokens_m'],'layer_ord':i+1,'gate_c':z.loc[i,'gate_c'],'path_up_norm_c':z.loc[i,'path_up_norm_c'],'path_down_norm_c':z.loc[i,'path_down_norm_c'],'next_effect_c':z.loc[i+1,'A_effect_magnitude_c']})
Lag=pd.DataFrame(lag);Lag.to_csv(OUT/'F3_lag_raw.csv',index=False)
Xl=standard(Lag[['gate_c','path_up_norm_c','path_down_norm_c']]);yl=standard(Lag[['next_effect_c']])[:,0]
bl,_,_,_,r2l=ols(Xl,yl)
# vectorized permutation within each 5-row checkpoint block
G3=Lag.checkpoint.nunique(); assert len(Lag)==G3*5
Y=yl.reshape(G3,5);B=5000;rng=np.random.default_rng(SEED+3)
perm=np.argsort(rng.random((B,G3,5)),axis=2)
YP=np.take_along_axis(np.broadcast_to(Y,(B,G3,5)),perm,axis=2).reshape(B,-1)
XtX_inv=np.linalg.inv(Xl.T@Xl);Xty=YP@Xl; Bet=Xty@XtX_inv.T
# y variance preserved by permutation; no intercept after standardization
sse=np.sum(YP*YP,axis=1)-np.sum(Bet*Xty,axis=1);sst=np.sum((YP-YP.mean(axis=1,keepdims=True))**2,axis=1);null=1-sse/sst
pd.DataFrame({'null_r2':null}).to_csv(OUT/'F3_lag_permutation_null.csv',index=False)
pperm=float((1+np.sum(null>=r2l))/(B+1))
summary['benchmarks']['F3']={'n_layer_rows':len(F3),'n_checkpoints':F3.checkpoint.nunique(),'centered_correlations':cor,'same_layer_regression':{'beta_gate':float(bs[0]),'beta_path_up':float(bs[1]),'beta_path_down':float(bs[2]),'r2':r2s},'lag_regression':{'beta_gate':float(bl[0]),'beta_path_up':float(bl[1]),'beta_path_down':float(bl[2]),'r2':r2l,'permutation_p':pperm,'null_r2_mean':float(null.mean()),'null_r2_95':ci(null)}}

# F4
F4=L[L.tokens_m.isin(COMPLETE)].copy();F4['D_minus_C']=F4.B_under_D_accuracy-F4.B_under_C_accuracy;F4.to_csv(OUT/'F4_C_vs_D_raw.csv',index=False)
a=F4.D_minus_C.to_numpy(float);pv={};pt={}
for v,z in F4.groupby('variant'):
 q=z.D_minus_C.to_numpy(float);pv[v]={'mean':float(q.mean()),'median':float(np.median(q)),'negative_fraction':float((q<0).mean()),'positive_fraction':float((q>0).mean()),'n':len(q)}
for t,z in F4.groupby('tokens_m'):
 q=z.D_minus_C.to_numpy(float);pt[str(int(t))]={'mean':float(q.mean()),'median':float(np.median(q)),'negative_fraction':float((q<0).mean()),'n':len(q)}
summary['benchmarks']['F4']={'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'ci95_cluster_checkpoint':cluster_boot_mean(F4,'D_minus_C',4000,SEED+4),'negative_fraction':float((a<0).mean()),'zero_fraction':float((a==0).mean()),'positive_fraction':float((a>0).mean()),'variants_with_negative_mean':sum(x['mean']<0 for x in pv.values()),'per_variant':pv,'per_token':pt}

# F5
fp=[]
for rep in D['full_layer_reports'].values():
    v=rep['variant_id'];t=int(rep['tokens_m'])
    if t not in COMPLETE:continue
    lay=rep['layers'];fin=lay['final'];ef=np.array([lay[x]['A_effect_magnitude'] for x in LAYERS],float);pk=int(np.argmax(ef));peak=float(ef[pk]);fe=float(ef[-1])
    r={'variant':v,'tokens_m':t,'A_effect_final':fe,'A_axis_final':float(fin['A_axis_consistency']),'B_condition_1_final':float(fin['B_condition_1_accuracy']),'B_under_C_final':float(fin['B_under_C_accuracy']),'B_under_D_final':float(fin['B_under_D_accuracy']),'negate_minus_zero_B':float(fin['B_intervention_accuracy_by_A_mode']['negate']-fin['B_intervention_accuracy_by_A_mode']['zero']),'shuffle_minus_zero_B':float(fin['B_intervention_accuracy_by_A_mode']['shuffle']-fin['B_intervention_accuracy_by_A_mode']['zero']),'B_cluster_margin_final':float(fin['B_cluster']['margin']),'C_relation_margin_final':float(fin['C_relative_to_B']['relation_margin']),'A_effect_peak_layer':pk+1,'A_effect_final_over_peak':float(fe/peak) if abs(peak)>1e-12 else np.nan}
    lr=fin.get('learned_A_reference')
    r['learned_B_ref_delta']=np.nan if lr is None else float(lr['relative_B_margin_by_reference']['learned']-(lr['relative_B_margin_by_reference']['shuffle']+lr['relative_B_margin_by_reference']['random'])/2)
    r['learned_C_ref_delta']=np.nan if lr is None else float(lr['relative_C_relation_margin_by_reference']['learned']-(lr['relative_C_relation_margin_by_reference']['shuffle']+lr['relative_C_relation_margin_by_reference']['random'])/2)
    fp.append(r)
FP=pd.DataFrame(fp);FP.to_csv(OUT/'F5_fingerprint_raw.csv',index=False)
subs={'intervention':['A_effect_final','A_axis_final','negate_minus_zero_B','shuffle_minus_zero_B','A_effect_peak_layer','A_effect_final_over_peak'],'reference':['learned_B_ref_delta','learned_C_ref_delta'],'depth_geometry_accessibility':['A_effect_peak_layer','A_effect_final_over_peak','B_condition_1_final','B_under_C_final','B_under_D_final','B_cluster_margin_final','C_relation_margin_final']}

def matrix(token,features):
 z=FP[FP.tokens_m==token].dropna(subset=features).sort_values('variant');lab=list(z.variant);A=z[features].to_numpy(float);sd=A.std(0);sd=np.where(sd<1e-12,1,sd);A=(A-A.mean(0))/sd;M=np.linalg.norm(A[:,None,:]-A[None,:,:],axis=2);return lab,M

def permcorr(e,l,features,B=10000,seed=0):
 le,A=matrix(e,features);ll,Z=matrix(l,features);common=sorted(set(le)&set(ll));ie=[le.index(x) for x in common];il=[ll.index(x) for x in common];A=A[np.ix_(ie,ie)];Z=Z[np.ix_(il,il)];n=len(common);tri=np.triu_indices(n,1);x=A[tri];y=Z[tri]
 obsP=corr(x,y,'pearson');obsS=corr(x,y,'spearman')
 rng=np.random.default_rng(seed+e*100+l);P=np.argsort(rng.random((B,n)),axis=1)
 # gather A[p][:,p] for all B
 AP=A[P[:,:,None],P[:,None,:]];XX=AP[:,tri[0],tri[1]]
 # row-wise Pearson
 yc=y-y.mean(); xc=XX-XX.mean(1,keepdims=True);den=np.sqrt(np.sum(xc*xc,1)*np.sum(yc*yc));rp=np.sum(xc*yc,1)/den
 # row-wise Spearman
 rx=rankdata(XX,axis=1);ry=rankdata(y);rxc=rx-rx.mean(1,keepdims=True);ryc=ry-ry.mean();den2=np.sqrt(np.sum(rxc*rxc,1)*np.sum(ryc*ryc));rs=np.sum(rxc*ryc,1)/den2
 return {'variants':common,'n_pairs':len(x),'spearman':obsS,'pearson':obsP,'spearman_perm_p_one_sided':float((1+np.sum(rs>=obsS))/(B+1)),'pearson_perm_p_one_sided':float((1+np.sum(rp>=obsP))/(B+1)),'spearman_null_95':ci(rs),'pearson_null_95':ci(rp)}
f5={}
for name,features in subs.items():f5[name]={f'{e}_to_120':permcorr(e,120,features,10000,SEED+10) for e in [25,50]}
summary['benchmarks']['F5']=f5

# exploratory E1, explicitly post-hoc
ex=[]
for rep in D['full_layer_reports'].values():
 v=rep['variant_id'];t=int(rep['tokens_m'])
 if t not in COMPLETE:continue
 lay=rep['layers'];bm=np.array([lay[x]['B_cluster']['margin'] for x in LAYERS]);cm=np.array([lay[x]['C_relative_to_B']['relation_margin'] for x in LAYERS]);ba=np.array([lay[x]['B_condition_1_accuracy'] for x in LAYERS])
 ex.append({'variant':v,'tokens_m':t,'checkpoint':checkpoint(v,t),'B_geometry_retention':float(bm[-1]/bm.max()) if bm.max()!=0 else np.nan,'C_geometry_retention':float(cm[-1]/cm.max()) if cm.max()!=0 else np.nan,'B_accuracy_final_minus_peak':float(ba[-1]-ba.max()),'B_accuracy_final':float(ba[-1]),'B_margin_final_minus_peak':float(bm[-1]-bm.max()),'C_margin_final_minus_peak':float(cm[-1]-cm.max())})
E=pd.DataFrame(ex);E.to_csv(OUT/'E1_geometry_function_dissociation_raw.csv',index=False)
summary['exploratory']={'E1':{'B_geom_retention_vs_accuracy_final_spearman':corr(E.B_geometry_retention,E.B_accuracy_final),'B_geom_loss_vs_accuracy_loss_spearman':corr(E.B_margin_final_minus_peak,E.B_accuracy_final_minus_peak),'C_geom_retention_vs_B_accuracy_final_spearman':corr(E.C_geometry_retention,E.B_accuracy_final)}}

(OUT/'benchmark_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
lines=['# Blind Follow-up Benchmark Machine Summary','']
for k,v in summary['benchmarks'].items():lines += [f'## {k}','```json',json.dumps(v,indent=2,sort_keys=True),'```','']
(OUT/'benchmark_summary.md').write_text('\n'.join(lines))
# hashes last, excluding itself
hr=[]
for p in sorted(OUT.glob('*')):
 if p.is_file() and p.name!='output_hashes.csv':hr.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
pd.DataFrame(hr).to_csv(OUT/'output_hashes.csv',index=False)
print(json.dumps(summary,indent=2,sort_keys=True))
