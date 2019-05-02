import os, json
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

# this finds our json files
path_to_json = 'C:\\Users\\lz826\\git\\learning-object-representations-by-mixing-scenes\\src\\final_model\\metrics\\test'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

metrics = pd.DataFrame(columns=['training_epoch', 'test_fid', 'test_is', 'train_fid', 'train_is'])

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)

        model_iteration = json_text['model_iteration']
        fid = json_text['fid']
        is_mean = json_text['is_mean']
        i = int(model_iteration)
        metrics.loc[i] = [model_iteration, fid, is_mean, None, None]

metrics = metrics.sort_values(by=['training_epoch'])
#print(metrics)

# metrics.plot()

# fids = metrics[['training_epoch', 'test_fid']]
# is_mean = metrics[['training_epoch', 'test_is']]

# fids.plot()
# ax = plt.gca()
# metrics.plot(kind='line', x='model_iteration', y='fid',ax=ax)
# metrics.plot(kind='line', x='model_iteration', y='is_mean',color="red",ax=ax)

# ax = metrics.plot(kind='scatter',x='model_iteration', y='fid', color='red')
# metrics.plot.line(x='model_iteration', y='fid', ax=ax, color='red')
metrics.plot.line(x='training_epoch', y='test_fid', style='-o')

# sns.lmplot("model_iteration", "fid", data=metrics, hue="is_mean", fit_reg=False, col='fid', col_wrap=2)
plt.show()




# plt.savefig('output.png')