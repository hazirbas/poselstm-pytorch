import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ
if opt.model == 'posenet':
    testepochs = numpy.arange(450, 500+1, 5)
else:
    testepochs = numpy.arange(450, 1200+1, 5)

testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch medX  medQ\n')
testfile.write('==================\n')

model = create_model(opt)
visualizer = Visualizer(opt)

for testepoch in testepochs:
    model.load_network(model.netG, 'G', testepoch)
    visualizer.change_log_path(testepoch)
    # test
    # err_pos = []
    # err_ori = []
    err = []
    print("epoch: "+ str(testepoch))
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()[0]
        print('\t%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
        image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
        pose = model.get_current_pose()
        visualizer.save_estimated_pose(image_path, pose)
        err_p, err_o = model.get_current_errors()
        # err_pos.append(err_p)
        # err_ori.append(err_o)
        err.append([err_p, err_o])

    median_pos = numpy.median(err, axis=0)
    if median_pos[0] < besterror[1]:
        besterror = [testepoch, median_pos[0], median_pos[1]]
    print()
    # print("median position: {0:.2f}".format(numpy.median(err_pos)))
    # print("median orientat: {0:.2f}".format(numpy.median(err_ori)))
    print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
    testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(testepoch,
                                                     median_pos[0],
                                                     median_pos[1]))
    testfile.flush()
print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('==================\n')
testfile.close()
