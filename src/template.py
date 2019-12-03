def set_template(args):
    # Set the templates here
    if args.template.find('DNLN') >= 0:
        args.data_train = 'Vimeo'
        args.data_test = 'Demo'


