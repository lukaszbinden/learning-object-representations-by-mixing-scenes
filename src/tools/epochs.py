import sys

# usage:  python epochs.py 100720 104121 6

if len(sys.argv) != 4:
    print('usage:  python epochs.py <current_iteration> <num_images_in_dataset> <batch_size>')
    sys.exit(0)

current_iteration = int(sys.argv[1])
num_images = int(sys.argv[2])
batch_size = int(sys.argv[3])

result = current_iteration / (num_images / batch_size)

print('epochs so far: %s' % (str(round(result, 4))))


