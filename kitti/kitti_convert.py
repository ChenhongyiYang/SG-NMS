'''
convert neural network output txt to kitti format
'''

import os

import argparse


KITTI_VEHICLES = ['Background', 'Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']
KITTI_CARS = ['Car', 'Truck', 'Van']

class Convertor(object):
    def __init__(self, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

    def read_in(self, in_txt):
        f = open(in_txt, 'r')
        lines = f.readlines()
        f.close()
        rois = []
        classes = []
        scores = []
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 6 or len(line) == 7:
                c = int(line[0])
                if c == 1:
                    ymin = float(line[1])
                    xmin = float(line[2])
                    ymax = float(line[3])
                    xmax = float(line[4])
                    score = float(line[5])
                    rois.append([ymin, xmin, ymax, xmax])
                    classes.append(c)
                    scores.append(score)
        return rois, classes, scores


    def write_out(self, out_txt, rois, classes, scores):
        f = open(out_txt, 'w')
        for i in range(len(classes)):
            f.write('%s '%KITTI_VEHICLES[classes[i]])
            f.write('-1 -1 -10 ')
            f.write('%d %d %d %d '%(rois[i][1], rois[i][0], rois[i][3], rois[i][2]))
            f.write('-1 -1 -1 -1000 -1000 -1000 -10 ')
            f.write('%.5f'%scores[i])
            f.write('\n')
        f.close()



    def run(self):
        in_list = os.listdir(self.in_dir)
        for in_txt in in_list:
            if in_txt.split('.')[-1] == 'txt':
                rois, classes, scores = self.read_in(os.path.join(self.in_dir, in_txt))
                self.write_out(os.path.join(self.out_dir,in_txt), rois, classes, scores)




if __name__ == '__main__':
    in_dir = '/home/grad3/hongyi/result_ae_nms/rfcn_baseline'
    out_dir = '/home/grad3/hongyi/result_ae_nms/rfcn_baseline_convert'

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    convertor = Convertor(in_dir, out_dir)
    convertor.run()
































