import torch.optim as Opt

def get_optimizer(model, args):

    # opt_fns = {
    #     'adam': Opt.Adam(model.parameters(), lr = args.lr_start),
    #     'sgd': Opt.SGD(model.parameters(), lr = args.lr_start),
    #     'adagrad': Opt.Adagrad(model.parameters(), lr = args.lr_start)
    # }
    # opt_fn = Opt.Adam(model.parameters(), lr=args.lr_start)
    opt_fn = Opt.Adam([{'params': model.parameters(),  'initial_lr': args.lr_start}], lr=args.lr_start)
    return opt_fn
