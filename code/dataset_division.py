import random
import os


def traverse_directory(directory, txt_name):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    file_names.sort()
    with open(txt_name, 'w') as f:
        for file_name in file_names:
            f.write(file_name + '\n')


def split_dataset(file_path, train_ratio=0.8, val_ratio=0.1, former=None):
    with open(file_path, 'r') as f:
        data = f.readlines()
    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_set = data[:train_size]
    val_set = data[train_size:train_size + val_size]
    test_set = data[train_size + val_size:]
    with open(former + '_trainset.txt', 'w') as f:
        f.writelines(train_set)
    with open(former + '_valset.txt', 'w') as f:
        f.writelines(val_set)
    with open(former + '_testset.txt', 'w') as f:
        f.writelines(test_set)
    print(f"Finished.")


if __name__ == "__main__":
    traverse_directory('JSRT-241/CXR', "JSRT.txt")
    split_dataset('JSRT.txt',0.8,0.1, "JSRT")
