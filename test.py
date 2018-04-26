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

# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

testepochs = ['latest']
testepochs = numpy.arange(150, 251, 5)
# testepochs = numpy.arange(255, 306, 5)
testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch medX medQ\n')
testfile.write('=================\n')

for testepoch in testepochs:
    opt.which_epoch = testepoch
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # test
    err_pos = []
    err_ori = []
    err = []
    print("epoch: "+ str(opt.which_epoch))
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()[0]
        print('%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
        image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
        pose = model.get_current_pose()
        visualizer.save_estimated_pose(image_path, pose)
        err_p, err_o = model.get_current_errors()
        err_pos.append(err_p)
        err_ori.append(err_o)
        err.append([err_p, err_o])

    print()
    print("median position: {0:.2f}".format(numpy.median(err_pos)))
    print("median orientat: {0:.2f}".format(numpy.median(err_ori)))
    testfile.write("{0:<5} {1:.2f} {2:.2f}\n".format(testepoch,
                                                      numpy.median(err_pos),
                                                      numpy.median(err_ori)))
    testfile.flush()
    del model
    del visualizer

testfile.close()
