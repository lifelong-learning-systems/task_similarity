import numpy as np


if __name__ == '__main__':
    with open('temp.txt') as f:
        lines = f.read().splitlines()

    split = [line.split(' ')[-1] for line in lines]
    split_x = [line.split(' ')[0] for line in lines]
    nums = [float(text) for text in split]
    nums_x = [int(text) for text in split_x]
    rounded = [round(num, 3) for num in nums]
    x = np.array(nums_x)
    y = np.array(rounded)
    print('Scatter points:')
    print('\t' + str(list(x)))
    print('\t' + str(list(y)))
    print('Log slopes:')
    log_y = np.log(y)
    log_x = np.log(x)
    log_slopes = [(log_y[i] - log_y[i-1])/(log_x[i] - log_x[i - 1]) for i in range(1, len(log_y))]
    print('\t' + str(log_slopes))