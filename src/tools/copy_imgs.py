'''
Author: LZ, 02.04.19, 03.05.19

Usage:
cd /home/lz01a008/src/logs/20190327_220759/metrics/fid/test/57
head filenames_test_20190327_220759_ep57.csv
# python copy_imges.py 000000417706_1,000000392900_1,000000029243_1,000000441336_1,000000082387_1,1000,img_mix_gen_3.png
python copy_imges.py 000000165516_1-000000001357_1-000000391129_1-000000542259_1-000000430990_1-0101-img_mix_gen_4149.png

Also see shortcut in ~/bin/fcp

'''
import sys
import os
import subprocess

# argv: 000000154169_1,000000270690_1,000000116214_1,000000383758_1,000000321895_1,1000,img_mix_gen_1.png
# OR argv: 000000165516_1-000000001357_1-000000391129_1-000000542259_1-000000430990_1-0101-img_mix_gen_4149.png

full_dir = "/home/lz01a008/git/learning-object-representations-by-mixing-scenes/src/datasets/coco/2017_test/version/v2/full"
log_dir = "20190327_220759"
exp = "exp73"
fid_dataset = "test"
epoch = "73"


dest = "/home/lz01a008/results/" + exp + "/test_images/" + fid_dataset + "/" + epoch + "/"
fid_dir = "/home/lz01a008/src/logs/" + log_dir + "/metrics/fid/" + fid_dataset + "/" + epoch + "/images/"
fmix_dir = "/home/lz01a008/src/logs/" + log_dir + "/metrics/fid/" + fid_dataset + "/" + epoch + "/images_all/mixed_feature/"

def copy(argv):
    # print(argv)
    assert len(argv) == 2, "just 2 args"

    file_str = argv[1]

    print("*********************************************************")
    print("ARGV..........:", file_str)
    print("full_dir......:", full_dir)
    print("log_dir.......:", log_dir)
    print("exp...........:", exp)
    print("fid_dataset...:", fid_dataset)
    print("epoch.........:", epoch)
    print("dest..........:", dest)
    print("fid_dir.......:", fid_dir)
    print("*********************************************************")


    tokens = file_str.split("-")
    dest_dir = dest + tokens[-1]
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print('created dest_dir: %s' % dest_dir)

    i = 1
    for token in tokens:
        if token == tokens[-2]:
            break

        src_name = token + ".jpg"
        src_dir = full_dir + "/" + src_name
        dest_name = str(i) + "_" + token + ".jpg"
        out_dir = dest_dir + "/" + dest_name

        cp(src_dir, out_dir, i)
        i += 1

    feature_mix_img_id = tokens[-1].split(".")[0].split("_")[-1]
    feature_mix_img = "img_mix_" + feature_mix_img_id + ".png"
    src_dir = fmix_dir + "/" + feature_mix_img
    out_dir = dest_dir + "/" + str(i) + "_" + feature_mix_img
    cp(src_dir, out_dir, i)
    i += 1

    src_dir = fid_dir + tokens[-1]
    mix_file = tokens[-2] + "_" + tokens[-1]
    out_dir = dest_dir + "/" + str(i) + "_" + mix_file
    cp(src_dir, out_dir, i)


def cp(src_dir, dest_dir, i):
    cmd = ['cp', src_dir, dest_dir]
    # cmd = ['echo', "hello from subprocess " + str(iteration)]
    print("[%d] cmd: %s..." % (i, cmd))
    process = subprocess.Popen(cmd)
    process.wait()
    print("[%d] done." % i)


if __name__ == '__main__':
    copy(argv=sys.argv)