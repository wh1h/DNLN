import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import os

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %.2f M'%(num_params/1000000))
    
def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            
            print('----------------------------------------------')
            print_network(_model)
            print('----------------------------------------------')
            
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


if __name__ == '__main__':
    main()
