def get_network(args):

    if 'mixer' in args.arch:
        from .pointmixer import getPointMixerSegNet as getNetwork
        # elif 'pointtrans' in args.arch:
        #     from .pointtrans import getPointTransSegNet as getNetwork
        # elif 'pointnet' == args.arch:
        #     from .pointnet   import getPointMixerSegNet as getNetwork
    else:
        raise NotImplementedError
    
    kwargs = \
        {
            'intraLayer': args.intraLayer,
            'interLayer': args.interLayer,
            'transup': args.transup,
            'transdown': args.transdown,
            'stride': args.downsample,
        }
    model = getNetwork(c=args.fea_dim, k=args.classes, nsample=args.nsample, **kwargs)

    return model
