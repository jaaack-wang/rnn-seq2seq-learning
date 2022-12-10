'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: Code for generating train, dev, and test sets 
for reversed strings and reduplicated strings. 
'''
import random
import itertools
from os.path import join
from string import ascii_lowercase
from utils import create_dir, save_ds_in_txt


max_num_per_seq_train = 250 
max_num_per_seq_dev = 500
max_num_per_seq_test = 1000
# Turns out that these seeds do not guarantee same data
# generated every time the script gets executed
random_seeds = [1623647, 3487852, 823747921]

# alphabet used to generate the data
alphabet = ascii_lowercase
# str length ranges for model training and in-distribution testing
in_distribution_ranges = [(11, 31), (41, 61)]
# str length ranges for out-of-distribution testing
out_distribution_ranges = [(1, 10), (31, 40), (61, 80)]


def reversed_str_pair(s):
    '''Returns a pair of reversed strings.'''
    return [s, s[::-1]]


def reduplicated_str_pair(s, n=1, hint=""):
    '''Returns a pair of reduplicated strings.
    
    Args:
        - s (str): the base form to reduplicate
        - n (int): num of times the based reduplicated.
                   defaults to 0. 
        - hint (str): indicator suffixed to the base to
                      indicate the num of reduplications. 
                      defaults to empty string. 
    '''
    return [s + hint * n, s * n]


def all_words_of_length(length, alphabet):
    return [''.join(list(b)) for b 
            in itertools.product(alphabet, repeat=length)]


def n_words_of_length(max_num_per_seq, 
                      length, alphabet):
    
    if max_num_per_seq >= pow(len(alphabet), length):
        return all_words_of_length(length, alphabet)

    out = set()
    while len(out) < max_num_per_seq:
        word = "".join(random.choices(alphabet, 
                                       k=length))
        out.add(word)
    
    return list(out)


def get_str_pairs(str_func, 
                  min_len=10, 
                  max_len=10, 
                  max_num_per_seq=1000, 
                  alphabet=ascii_lowercase, 
                  shuffle=False):
    out = []
    for n in range(min_len, max_len+1):
        for s in n_words_of_length(
            max_num_per_seq, n, alphabet):
            
            out.append(str_func(s))
    
    if shuffle:
        random.shuffle(out)
    return out


def generate_data(str_func, train_size, dev_size, test_size, alphabet):
    
    split_1 = train_size
    split_2 = train_size + dev_size
    train, dev, test = [], [], []
    total = sum((train_size, dev_size, test_size))
    for (l, h) in in_distribution_ranges:
        for n in range(l, h):
            out = get_str_pairs(str_func, n, n,
                                total, alphabet, True)
            train.extend(out[:split_1])
            dev.extend(out[split_1:split_2])
            test.extend(out[split_2:])

    out_dist_test = []
    for (mi, mx) in out_distribution_ranges:
        out_dist_test.extend(get_str_pairs(
                    str_func, mi, mx, test_size))
        
    return train, dev, test, out_dist_test


def main():
    root = "../data"
    create_dir(join(root, "RevStr"))
    create_dir(join(root, "RedStr"))
    str_names = ["RevStr", "RedStr"]

    str_funcs = [reversed_str_pair, reduplicated_str_pair]
    for str_name, str_func in zip(str_names, str_funcs):
        for idx, seed in enumerate(random_seeds):
            
            random.seed(seed)
            train, dev, test, out_dist_test = generate_data(str_func, 
                                                            max_num_per_seq_train, 
                                                            max_num_per_seq_dev, 
                                                            max_num_per_seq_test,
                                                            alphabet)
            n = idx+1
            fp_prefix = join(root, str_name, str(n))
            create_dir(fp_prefix)
            save_ds_in_txt(train, join(fp_prefix, "train.txt"))
            save_ds_in_txt(dev, join(fp_prefix, "dev.txt"))
            save_ds_in_txt(test, join(fp_prefix, "test.txt"))
            save_ds_in_txt(out_dist_test, join(fp_prefix, "test2.txt"))
            
            
if __name__ == "__main__":
    main()
