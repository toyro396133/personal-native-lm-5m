import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
BASE=Path('/mnt/data/followup_work/results')
DATA=json.load(open('/mnt/data/followup_work/dataset.json'))
L=pd.read_csv(BASE/'all_primitive_layer_rows.csv')
COMP=[25,50,75,100,110,120]
LAYERS=['layer_1','layer_2','layer_3','layer_4','layer_5','layer_6','final']

def ols(X,y):
 X=np.asarray(X,float);y=np.asarray(y,float);b=np.linalg.lstsq(X,y,rcond=None)[0];r=y-X@b;return b,float(r@r)
def corr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
 return float(spearmanr(a[m],b[m]).statistic) if m.sum()>=3 and np.std(a[m])>1e-12 and np.std(b[m])>1e-12 else float('nan')

out={}
# E2 per-variant F1 decomposition
raw=[]
for v in [f'V{i}' for i in range(2,8)]:
 z=L[(L.variant==v)&L.tokens_m.isin(COMP)].copy()
 rows=[]
 for _,r in z.iterrows():
  g=f"{r['checkpoint']}|{r['layer']}"
  for mode in ['zero','negate','shuffle']:
   rows.append({'group':g,'mode':mode,'acc':r[f'B_A_{mode}_accuracy'],'norm':r[f'A_{mode}_effect_norm']})
 d=pd.DataFrame(rows)
 for c in ['acc','norm']:d[c+'_c']=d[c]-d.groupby('group')[c].transform('mean')
 d['neg']=(d['mode']=='negate').astype(float);d['shu']=(d['mode']=='shuffle').astype(float)
 for c in ['neg','shu']:d[c+'_c']=d[c]-d.groupby('group')[c].transform('mean')
 b0,s0=ols(d[['norm_c']],d.acc_c);b1,s1=ols(d[['norm_c','neg_c','shu_c']],d.acc_c)
 raw.append({'variant':v,'beta_norm':b1[0],'beta_negate_residual':b1[1],'beta_shuffle_residual':b1[2],'sse_improvement_fraction':(s0-s1)/s0,
             'mean_negate_minus_zero_acc':float((z.B_A_negate_accuracy-z.B_A_zero_accuracy).mean()),
             'mean_shuffle_minus_zero_acc':float((z.B_A_shuffle_accuracy-z.B_A_zero_accuracy).mean()),
             'mean_negate_minus_zero_norm':float((z.A_negate_effect_norm-z.A_zero_effect_norm).mean())})
E2=pd.DataFrame(raw);E2.to_csv(BASE/'E2_F1_per_variant_posthoc.csv',index=False);out['E2_F1_per_variant']=raw

# E3 reference specificity developmental table
F2=pd.read_csv(BASE/'F2_reference_specificity_raw.csv')
cols=['variant','tokens_m','B_learned_vs_mean_alt','C_learned_vs_mean_alt']
E3=F2.groupby(['variant','tokens_m'])[['B_learned_vs_mean_alt','C_learned_vs_mean_alt']].mean().reset_index()
E3.to_csv(BASE/'E3_reference_specificity_by_variant_token.csv',index=False)
out['E3_reference_specificity_by_variant_token']=E3.to_dict(orient='records')

# E4 all-pairs fingerprint trajectory matrix
FP=pd.read_csv(BASE/'F5_fingerprint_raw.csv')
subs={'intervention':['A_effect_final','A_axis_final','negate_minus_zero_B','shuffle_minus_zero_B','A_effect_peak_layer','A_effect_final_over_peak'],
      'reference':['learned_B_ref_delta','learned_C_ref_delta'],
      'depth_geometry_accessibility':['A_effect_peak_layer','A_effect_final_over_peak','B_condition_1_final','B_under_C_final','B_under_D_final','B_cluster_margin_final','C_relation_margin_final']}
def distvec(t,feat):
 z=FP[FP.tokens_m==t].dropna(subset=feat).sort_values('variant');lab=list(z.variant);A=z[feat].to_numpy(float);sd=A.std(0);sd=np.where(sd<1e-12,1,sd);A=(A-A.mean(0))/sd;M=np.linalg.norm(A[:,None,:]-A[None,:,:],axis=2);tri=np.triu_indices(len(z),1);return lab,M[tri]
pairs=[]
for name,feat in subs.items():
 for i,a in enumerate(COMP):
  for b in COMP[i+1:]:
   la,x=distvec(a,feat);lb,y=distvec(b,feat)
   # same eligible labels across complete tokens per subset
   pairs.append({'subset':name,'token_a':a,'token_b':b,'spearman_distance_geometry':corr(x,y)})
E4=pd.DataFrame(pairs);E4.to_csv(BASE/'E4_fingerprint_all_token_pairs_posthoc.csv',index=False);out['E4_fingerprint_all_pairs']=pairs

# E5 vector norm vs effect across training
rows=[]
for rep in DATA['full_layer_reports'].values():
 v=rep['variant_id'];t=int(rep['tokens_m']);diag=rep.get('A_mechanism_diagnostics')
 if v=='V1' or t not in COMP or not diag:continue
 effects=np.array([rep['layers'][x]['A_effect_magnitude'] for x in LAYERS[:6]],float)
 rows.append({'variant':v,'tokens_m':t,'A_vector_norm':diag['A_vector_norm'],'mean_internal_A_effect':effects.mean(),'final_A_effect':rep['layers']['final']['A_effect_magnitude']})
E5=pd.DataFrame(rows)
byv=[]
for v,z in E5.groupby('variant'):
 byv.append({'variant':v,'rho_vector_vs_internal_effect':corr(z.A_vector_norm,z.mean_internal_A_effect),'rho_vector_vs_final_effect':corr(z.A_vector_norm,z.final_A_effect)})
# within-variant centered pooled correlations
for c in ['A_vector_norm','mean_internal_A_effect','final_A_effect']:E5[c+'_c']=E5[c]-E5.groupby('variant')[c].transform('mean')
pooled={'rho_centered_vector_vs_internal':corr(E5.A_vector_norm_c,E5.mean_internal_A_effect_c),'rho_centered_vector_vs_final':corr(E5.A_vector_norm_c,E5.final_A_effect_c)}
E5.to_csv(BASE/'E5_A_vector_norm_vs_effect_posthoc.csv',index=False);out['E5_A_vector_norm_vs_effect']={'per_variant':byv,'pooled':pooled}

# E6 partial 130 descriptive only
z130=L[L.tokens_m==130].copy()
# construct reference deltas from columns
for targ in ['B','C']:
 z130[f'{targ}_learned_delta']=z130[f'{targ}_ref_learned_margin']-(z130[f'{targ}_ref_shuffle_margin']+z130[f'{targ}_ref_random_margin'])/2
z130['D_minus_C']=z130.B_under_D_accuracy-z130.B_under_C_accuracy
E6=z130.groupby('variant').agg(B_learned_delta=('B_learned_delta','mean'),C_learned_delta=('C_learned_delta','mean'),D_minus_C=('D_minus_C','mean')).reset_index()
# 120 counterparts same variants
z120=L[(L.tokens_m==120)&L.variant.isin(E6.variant)].copy()
for targ in ['B','C']:
 z120[f'{targ}_learned_delta']=z120[f'{targ}_ref_learned_margin']-(z120[f'{targ}_ref_shuffle_margin']+z120[f'{targ}_ref_random_margin'])/2
z120['D_minus_C']=z120.B_under_D_accuracy-z120.B_under_C_accuracy
C120=z120.groupby('variant').agg(B_learned_delta_120=('B_learned_delta','mean'),C_learned_delta_120=('C_learned_delta','mean'),D_minus_C_120=('D_minus_C','mean')).reset_index()
E6=E6.merge(C120,on='variant');E6['C_sign_same_120_130']=np.sign(E6.C_learned_delta)==np.sign(E6.C_learned_delta_120);E6['D_sign_same_120_130']=np.sign(E6.D_minus_C)==np.sign(E6.D_minus_C_120)
E6.to_csv(BASE/'E6_partial_130_descriptive.csv',index=False);out['E6_partial_130']=E6.to_dict(orient='records')

# E7 F4 leave-one-variant and per-layer stress
F4=pd.read_csv(BASE/'F4_C_vs_D_raw.csv')
loo=[]
for v in sorted(F4.variant.unique()):
 q=F4[F4.variant!=v].D_minus_C
 loo.append({'left_out':v,'mean':float(q.mean()),'negative_fraction':float((q<0).mean())})
pl=F4.groupby('layer').D_minus_C.agg(['mean','median']).reset_index();pl['negative_fraction']=F4.groupby('layer').D_minus_C.apply(lambda x:(x<0).mean()).values
pl.to_csv(BASE/'E7_D_vs_C_layer_stress.csv',index=False);out['E7_D_vs_C_stress']={'leave_one_variant_out':loo,'per_layer':pl.to_dict(orient='records')}

Path(BASE/'exploratory_counterchecks.json').write_text(json.dumps(out,indent=2,allow_nan=True))
print(json.dumps(out,indent=2,allow_nan=True))
