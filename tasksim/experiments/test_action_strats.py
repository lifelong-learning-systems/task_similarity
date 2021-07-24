import tasksim
import tasksim.gridworld_generator as gen

if __name__ == '__main__':
    grid = gen.create_grid((3, 3))
    grid[0][0] = 1