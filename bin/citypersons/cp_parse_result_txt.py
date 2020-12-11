import os
import numpy as np 


def parse_txt(result_txt):
    f = open(result_txt, 'r')
    lines = f.readlines()
    f.close()

    dirs = []
    mrs_all = []

    for i, line in enumerate(lines):
        if i % 2 == 0:
            dirs.append(line.strip())
        else:
            mrs = [float(c) for c in line.strip().split()]
            mrs_all.append(mrs)
    return dirs, mrs_all



def show_top(result_txt, top=5):
    names = ['Reasonable', 'Reasonable_small','Reasonable_occ=heavy', 'All', 'Reasonable', 'bare', 'partial', 'heavy']
    dirs, mrs_all = parse_txt(result_txt)
    np_mrs_all = np.array(mrs_all)
    
    M = len(mrs_all[0])

    for i in range(M):
        print(names[i])
        mrs_i = np_mrs_all[:, i]
        top_inds = np.argsort(mrs_i)[:top]
        for j in range(top):
            print(dirs[top_inds[j]], np_mrs_all[top_inds[j]])
        print('\n\n\n')


if __name__ == '__main__':
    txt  = '/scratch/ChenhongyiYang/cp_result/ags_2.txt'
    show_top(txt)






































