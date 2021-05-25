
# 2 -> 11
sizes = list(range(2, 12))

success_prob = 0.9
for sz in sizes:
    with open(f'gridworlds/{sz}x{sz}_base.txt', 'w+') as f:
        f.write(f'{success_prob}\n')
        f.write(f'{sz} {sz}\n')
        center = sz // 2
        for i in range(sz):
            for j in range(sz):
                char = '2' if i == center and j == center else '0'
                f.write(char)
            f.write('\n')
