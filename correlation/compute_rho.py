import pandas as pd
import pickle

path = 'result/sss_cf10_meco_opt.p'
# path = 'nb2_sss_cf10_seed42_dlappoint_dlinfo1_initwnone_initbnone_1.p'
meco = []
accs = []
with open(path, 'rb') as f:
    while True:
        try:
            fl = pickle.load(f)
            meco.append(fl['meco'][0])
            accs.append(fl['testacc'])
        except:
            break

N = len(meco)
print(N)
df = pd.DataFrame({
    'meco':meco[:N],
    'acc': accs[:N]
    })
print(df.corr(method='spearman'))

