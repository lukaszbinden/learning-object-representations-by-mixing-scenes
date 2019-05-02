import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth_tb(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def main():
    # this finds our json files
    metrics_test_path = 'C:\\Users\\lz826\\git\\learning-object-representations-by-mixing-scenes\\src\\final_model\\metrics\\test'
    metrics_training_path = 'C:\\Users\\lz826\\git\\learning-object-representations-by-mixing-scenes\\src\\final_model\\metrics\\training'
    test_json_files = [pos_json for pos_json in os.listdir(metrics_test_path) if pos_json.endswith('.json')]
    training_json_files = [pos_json for pos_json in os.listdir(metrics_training_path) if pos_json.endswith('.json')]

    metrics = pd.DataFrame(columns=['training_epoch', 'test_fid', 'test_fid_sm', 'test_is', 'train_fid', 'train_is'])
    test_fids = pd.DataFrame(columns=['training_epoch', 'test_fid'])

    iter_to_values = {}
    for index, js in enumerate(test_json_files):
        with open(os.path.join(metrics_test_path, js)) as json_file:
            json_text = json.load(json_file)

            model_iteration = json_text['model_iteration']
            fid = json_text['fid']
            is_mean = json_text['is_mean']
            i = int(model_iteration)
            iter_to_values[i] = {'training_epoch': model_iteration, 'test_fid':fid, 'test_is': is_mean, 'train_fid':None, 'train_is':None}
            test_fids.loc[i] = [model_iteration, fid]

    for index, js in enumerate(training_json_files):
        with open(os.path.join(metrics_training_path, js)) as json_file:
            json_text = json.load(json_file)

            model_iteration = json_text['model_iteration']
            fid = json_text['fid']
            is_mean = json_text['is_mean']
            i = int(model_iteration)
            if i in iter_to_values:
                iter_to_values[i]['train_fid'] = fid
                iter_to_values[i]['train_is'] = is_mean
            else:
                print("training model_iteration=%d not in test! skip..." % model_iteration)

    print(test_fids)
    test_fids = test_fids.sort_values(by=['training_epoch'], ascending=True)
    # print(test_fids['test_fid'].tolist())
    # test_fid_sm = smooth(test_fids['test_fid'], 19)
    test_fid_list = test_fids['test_fid'].tolist()
    test_fid_sm = smooth_tb(test_fid_list, 0.9)
    # print("test_fid_list: ", test_fid_list)
    # print("test_fid_sm: ", test_fid_sm)

    id = 0
    for key in iter_to_values.keys():
        if key > 88:
            break # abort because of missing data...
        if iter_to_values[key]['train_fid']:
            metrics.loc[key] = [iter_to_values[key]['training_epoch'], iter_to_values[key]['test_fid'], test_fid_sm[id], iter_to_values[key]['test_is'], iter_to_values[key]['train_fid'], iter_to_values[key]['train_is']]
        else:
            print("%s epcoh does not have training values, skip it..." % iter_to_values[key]['training_epoch'])
        id += 1


    metrics = metrics.sort_values(by=['training_epoch'])
    #print(metrics)

    # metrics.plot()

    # fids = metrics[['training_epoch', 'test_fid']]
    # is_mean = metrics[['training_epoch', 'test_is']]

    # fids.plot()
    ax = plt.gca()
    # metrics.plot(kind='line', x='model_iteration', y='fid',ax=ax)
    # metrics.plot(kind='line', x='model_iteration', y='is_mean',color="red",ax=ax)

    # ax = metrics.plot(kind='scatter',x='model_iteration', y='fid', color='red')
    # metrics.plot.line(x='model_iteration', y='fid', ax=ax, color='red')
    mplot = metrics.plot.line(x='training_epoch', y='test_fid', ax=ax, color='b')
    metrics.plot.line(x='training_epoch', y='train_fid', ax=ax, color='orange')
    metrics.plot.line(x='training_epoch', y='test_fid_sm', color="r", ax=ax)

    mplot.set_xlabel("Training Epoch")
    mplot.set_ylabel("FID")
    mplot.legend(["Test", "Training", "Test smoothed"])

    # sns.lmplot("model_iteration", "fid", data=metrics, hue="is_mean", fit_reg=False, col='fid', col_wrap=2)
    plt.show()




    # plt.savefig('output.png')


if __name__ == '__main__':
    main()