
"""
    This experiment used to verify the two subset if consistent when calculate the DAD score
    using different models.
"""

from trainer.utils import DataMap
import numpy as np
import matplotlib.pyplot as plt
from params import params_msd_nih_for_hard_example_delete as params_file
args = params_file.Params.trainparam()

# q = 0.2
q_ = np.arange(0.05, 1, 0.02)
overlap_percent = []
for q in q_:
    x = np.load('/data/HYK/DMDS_save/msd_nih/exp5-GraNd/No_Prunning/dice_files/i7236_e200_patch_dice.npy', allow_pickle=True).item()
    datamap = DataMap(args, save_path='/data/HYK/DMDS_save/msd/exp7-GraNd/No_Prunning/data_maps', dice_file=x)
    wait_to_del = datamap.data_select(q=q, mode='easy', score_mode='datamap')
    dataset = list(x.keys())
    subset_VNet = []
    for data in dataset:
        if data not in wait_to_del:
            subset_VNet.append(data)


    x0 = np.load('/data/MING/data/AUTODOL/result/dice_files/e50_patch_dice_trainMSD_NIHtrain_limited.npy', allow_pickle=True).item()
    # x = {}
    # for i in dataset:
    #     x[i] = x0[i]

    for i in list(x0.keys()):
        if i not in list(x.keys()):
            del x0[i]

    datamap = DataMap(args, save_path='/data/HYK/DMDS_save/msd/exp7-GraNd/No_Prunning/data_maps', dice_file=x0)
    wait_to_del0 = datamap.data_select(q=q, mode='easy', score_mode='datamap')
    dataset = list(x.keys())
    subset_Swin = []
    for data in dataset:
        if data not in wait_to_del0:
            subset_Swin.append(data)

    overlap = 0
    for data in dataset:
        if data in subset_VNet and data in subset_Swin:
            overlap += 1

    overlap = overlap/len(subset_Swin)
    overlap_percent.append(overlap)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(q_, overlap_percent, color='cornflowerblue')
ax.xaxis.set_ticks(np.arange(0.05, 0.96, 0.15))
plt.title('V-Net vs. SwinUNETR, Prund Examples with DAD')
plt.ylabel('Percent Overlap')
plt.xlabel('Fraction of Dataset Pruned')
plt.grid()
plt.xlim([0.05, 0.95])
plt.ylim([0.35, 1])
# plt.show()
plt.savefig('/data/HYK/DMDS_save/figure/different model architectures/different model architectures.pdf')
