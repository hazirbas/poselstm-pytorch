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
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
err_pos = []
err_ori = []
err = []
for i, data in enumerate(dataset):
    # if i >= opt.how_many:
    #     break
    model.set_input(data)
    model.test()
    # visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path), end='\r')
    # visualizer.save_images(webpage, visuals, img_path)
    err_p, err_o = model.get_current_errors()
    err_pos.append(err_p)
    err_ori.append(err_o)
    err.append([err_p, err_o])

# webpage.save()
print(numpy.median(err, axis=0))
print("median position: {0:.2f}".format(numpy.median(err_pos)))
print("median orientat: {0:.2f}".format(numpy.median(err_ori)))
