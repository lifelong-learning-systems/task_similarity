from matplotlib import pyplot as plt

RESULTS_DIR='results_transfer'

file_names = ['new_res.txt', 'song_res.txt']

for f in file_names:
    f2 = f'{RESULTS_DIR}/{f}'
    with open(f2) as ptr:
        lines = ptr.readlines()
    
    def process_line(line):
        line = line.rstrip()
        line = line[1:-1]
        tokens = line.split(', ')
        return [float(token) for token in tokens]
    y = process_line(lines[0])
    x = process_line(lines[1])
    plt.clf()
    plt.scatter(x, y)
    plt.show()