import numpy as np
import matplotlib.pyplot as plt

N = 4
men_means = (34.6, 33.8, 35.7, 26.3)
# men_std = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
ind = np.array([0,0.5,1,1.5])
width = 0.25       # the width of the bars

fig, ax = plt.subplots(1,1,figsize=(4,3))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
rects1 = ax.bar(ind, men_means, width, color='b')

# women_means = (25, 32, 34, 20)

# add some text for labels, title and axes ticks
ax.set_ylabel('STL-10 Accuracy %')
ax.set_xlabel('Ablation experiment')
# ax.set_title('STL-10 test set accuracy')
ax.set_xticks(ind)
ax.set_xticklabels((r'Full', r'$-\mathcal{L}_{rec}$', r'$-\mathcal{L}^G_{adv}$', r'$-\mathcal{L}_{Cls}$'))

# ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        print(height)
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%s' % str(height),
                ha='center', va='bottom')

autolabel(rects1)

plt.show()