
import matplotlib.pyplot as plt
import os
import numpy as np
from trainer.utils import DataMap
from params import params_msd_only_for_hard_example_delete as params_file
args = params_file.Params.trainparam()


def entropy(root, es, et, ei, gap):

    dice_path = os.listdir(root)
    dice_path.sort()
    for t in range(10, gap, 10):
        E0, E1, E2, E3 = [], [], [], []
        for e in range(es, et, ei):
            for i in dice_path:
                if f'e{e}_' in i:
                    x = np.load(f'{root}/{i}', allow_pickle=True).item()
                    d = DataMap(args, '...', dice_file=x)
                    confidence, variability, correctness = d.statistics()
                    break
            for i in dice_path:
                if f'e{e + t}_' in i:
                    x1 = np.load(f'{root}/{i}', allow_pickle=True).item()
                    d1 = DataMap(args, '...', dice_file=x1)
                    confidence1, variability1, correctness1 = d1.statistics()
                    break

            conf = confidence1 - confidence
            var = variability1 - variability

            index = np.argsort(conf)
            conf1, var1 = conf.copy(), var.copy()

            name = list(x.keys())
            # 移动方向
            area1, area2, area3, area4, move_distance = [], [], [], [], []
            # 各个移动方向的平均移动距离
            area1_md, area2_md, area3_md, area4_md = [], [], [], []
            for i in range(len(conf1)):
                l = np.absolute(np.array([conf1[i], var1[i]]))
                if conf1[i] > 0 and var1[i] > 0:
                    area1.append(name[i])
                    area1_md.append(l)
                elif conf1[i] > 0 and var1[i] < 0:
                    area2.append(name[i])
                    area2_md.append(l)
                elif conf1[i] < 0 and var1[i] < 0:
                    area3.append(name[i])
                    area3_md.append(l)
                elif conf1[i] < 0 and var1[i] > 0:
                    area4.append(name[i])
                    area4_md.append(l)
                # move_distance.append(l)
                move_distance.append(conf1[i]+var1[i])

            total = len(conf)
            p1, p2, p3, p4 = len(area1) / total, len(area2) / total, len(area3) / total, len(area4) / total
            p1, p2, p3, p4 = p1+1e-3, p2+1e-3, p3+1e-3, p4+1e-3
            li = [1e-3, 1e-3, 1e-3, 1e-3]
            mdi = [area1_md, area2_md, area3_md, area4_md]
            for i in range(4):
                if mdi[i]:
                    li[i] = np.mean(mdi[i])
            l1, l2, l3, l4 = li

            L = np.sum(move_distance)

            s = 1/(1+np.exp((e-690/2)/345))

            E = -(p1 * np.log(p1) + p2 * np.log(p2) +
                  p3 * np.log(p3) + p4 * np.log(p4))

            E0.append(E)
            E1.append(
                -(p1 * np.log(p1) + p2 * np.log(p2) +
                  p3 * np.log(p3) + p4 * np.log(p4))*s/L)
            E2.append(
                -(l1/p1 / np.log(p1) + l2/p2 / np.log(p2) +
                  l3/p3 / np.log(p3) + l4/p4 / np.log(p4))/s)
            E3.append(L)

        plt.subplot(3, 1, 1)
        plt.plot(list(range(es, et, ei)), E0)
        plt.subplot(3, 1, 2)
        plt.plot(list(range(es, et, ei)), E1)
        plt.subplot(3, 1, 3)
        plt.plot(list(range(es, et, ei)), E3)

    max_E = -(np.log(0.25)) * np.ones(int((et - es) / 10))
    # plt.plot(list(range(es, et, ei)), max_E, '--', c='darkred')
    # plt.plot(list(range(es, et, ei)), max_E * 0.9, '--', c='orange')
    # plt.plot(list(range(es, et, ei)), max_E * 0.8, '--', c='green')
    # plt.text(5, 1.12, r'$0.8E_{max}$', fontsize=12)
    # plt.text(5, 1.26, r'$0.9E_{max}$', fontsize=12)
    # plt.text(5, 1.4, r'$E_{max}$', fontsize=12)
    # plt.ylim(0, 1.45)
    plt.xlim(0, 600)
    plt.xlabel('epoch')
    plt.legend(['e10'])
    # plt.savefig('/data/HYK/DMDS_save/figure/{dataset}_entropy.jpg')
    plt.close()
    return E3



task = 'MSD_NIH'
root = '/data/HYK/DMDS_save/msd_nih/exp2-GraNd/No_Prunning/dice_files'

es, et, ei = 10, 300, 10
gap = 40


list_path = os.listdir(root)
list_path.sort()
total_x = {}
for j in range(len(list_path)):
    if f'e10_' in list_path[j]:
        total_x = np.load(f'{root}/{list_path[j]}', allow_pickle=True).item()
for i in range(20, 300, 10):
    for j in range(len(list_path)):
        if f'e{i}_' in list_path[j]:
            x = np.load(f'{root}/{list_path[j]}', allow_pickle=True).item()
            for n in list(total_x.keys()):
                total_x[n]['dice'] = np.append(total_x[n]['dice'], x[n]['dice'])


d1_t = {'Ours': [0.7353, 0.7428, 0.7664, 0.7636, 0.7688, 0.7663, 0.7642],
        'axis': [10, 50, 100, 180, 200, 300, 400]
        }
d2_t = {'Ours': [0.7768, 0.7136, 0.7577, 0.7681, 0.7768, 0.7689, 0.7636],
        'axis': [10, 50, 100, 180, 200, 300, 400]
        }
d3_t = {'Ours': [0.7102, 0.68, 0.688, 0.6977, 0.7052, 0.7, 0.7052],
        'axis': [10, 50, 100, 170, 200, 300, 400]
        }



mode = 'hard'
score = 'datamap'
root = '/data/HYK/DMDS_save/msd_nih/exp2-GraNd/No_Prunning/dice_files'

es, et, ei = 10, 500, 10
gap = 20

E = entropy(root, es, et, ei, gap)

E_t = {'Ours': E,
        'axis': list(range(0, len(E)*10, 10))
        }
# d1_t['L'] = [E[0], E[5], E[10], E[20], E[30], E[40]]
d = [d1_t]

fig = plt.figure(figsize=(3.1, 3))

ax1 = fig.add_subplot(111)
plt.grid()
lin1 = ax1.plot(d[0]['axis'], d[0]['Ours'], c='steelblue', linewidth=3, label='Dice')
# ax1.set_ylabel('Dice', fontdict={'weight': 'normal', 'size': 12})
ax1.axis([0, 400, 0.7, 0.8])

ax2 = ax1.twinx()
lin2 = ax2.plot(list(range(10, len(E)*10, 10)), E[1:], c='lightcoral', linewidth=3, label='L')
# ax2.set_ylabel('moving distance', fontdict={'weight': 'normal', 'size': 12})
lns = lin1+lin2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs)
t = 0
for i, j in enumerate(E[1:]):
    if j <= max(E[1:])*0.01:
        plt.scatter([10, i*10], [max(E[1:]), j], zorder=10, color='forestgreen')
        break
# plt.xlabel('Score Computation Epoch')
# plt.show()
plt.savefig('/data/HYK/DMDS_save/figure/msd.png')


root = '/data/HYK/DMDS_save/nih_word/dice_files'

es, et, ei = 10, 500, 10
gap = 20

E = entropy(root, es, et, ei, gap)

E_t = {'Ours': E,
        'axis': list(range(10, len(E)*10, 10))
        }
d1_t['L'] = [E[0], E[5], E[10], E[20], E[30], E[40]]
d = [d2_t]

fig = plt.figure(figsize=(3.1, 3))
ax1 = fig.add_subplot(111)
plt.grid()
lin1 = ax1.plot(d[0]['axis'], d[0]['Ours'], c='steelblue', linewidth=3, label='Dice')
# ax1.set_ylabel('Dice', fontdict={'weight': 'normal', 'size': 12})
ax1.axis([0, 400, 0.7, 0.8])

ax2 = ax1.twinx()
lin2 = ax2.plot(list(range(10, len(E)*10, 10)), E[1:], c='lightcoral', linewidth=3, label='moving distance')
# ax2.set_ylabel('moving distance', fontdict={'weight': 'normal', 'size': 12})
lns = lin1+lin2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs)
t = 0
for i, j in enumerate(E[1:]):
    if j <= max(E[1:])*0.01:
        plt.scatter([30, i*10+10], [max(E[1:]), j], zorder=10, color='forestgreen')
        break
# plt.show()
plt.savefig('/data/HYK/DMDS_save/figure/nih.png')


root = '/data/HYK/DMDS_save/msd_word/exp4-GraNd/No_Prunning/dice_files'

es, et, ei = 10, 500, 10
gap = 20

E = entropy(root, es, et, ei, gap)

E_t = {'Ours': E,
        'axis': list(range(10, len(E)*10, 10))
        }
d1_t['L'] = [E[0], E[5], E[10], E[20], E[30], E[40]]
d = [d3_t]

fig = plt.figure(figsize=(3.1, 3))
ax1 = fig.add_subplot(111)
plt.grid()
lin1 = ax1.plot(d[0]['axis'], d[0]['Ours'], c='steelblue', linewidth=3, label='Dice')
# ax1.set_ylabel('Dice', fontdict={'weight': 'normal', 'size': 12})
ax1.axis([0, 400, 0.6, 0.75])

ax2 = ax1.twinx()
lin2 = ax2.plot(list(range(10, len(E)*10, 10)), E[1:], c='lightcoral', linewidth=3, label='moving distance')
# ax2.set_ylabel('moving distance', fontdict={'weight': 'normal', 'size': 12})
lns = lin1+lin2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs)
t = 0
for i, j in enumerate(E[1:]):
    if j <= max(E[1:])*0.01:
        plt.scatter([10, i*10+10], [max(E[1:]), j], zorder=10, color='forestgreen')
        break
# plt.show()
plt.savefig('/data/HYK/DMDS_save/figure/word.png')
