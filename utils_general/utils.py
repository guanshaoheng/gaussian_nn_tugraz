import os
import pickle


def writeLine(fname, s, mode='w'):
    f = open(fname, mode=mode)
    f.write(s)
    f.close()


def echo(*args):
    print('\n' + '=' * 80)
    for i in args:
        print('\t%s' % i)


def check_mkdir(*args):
    for path in args:
        if not os.path.exists(path):
            os.mkdir(path)
            print('\t\tDirectory made as %s' % path)


def pickle_dump(**kwargs):
    root_path = kwargs['root_path']
    savePath = os.path.join(root_path, 'scalar')
    check_mkdir(savePath)
    for k in kwargs:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'wb')
            pickle.dump(kwargs[k], f, 0)
            f.close()
    print('\tScalar saved in %s' % savePath)


def pickle_load(*args, root_path):
    cwd = os.getcwd()
    # if 'FEMxML' not in cwd:
    #     root_path = os.path.join(cwd, 'FEMxML')
    if 'sciptes4figures' in cwd:
        root_path = os.getcwd()
    savePath = os.path.join(root_path, 'scalar')
    # if not os.path.exists(savePath):
    #     os.mkdir(savePath)
    if 'epoch' in root_path:
        root_path = os.path.split(root_path)[0]
        savePath = os.path.join(root_path, 'scalar')
    print()
    print('-' * 80)
    print('Note: Scalar restored from %s' % savePath)
    for k in args:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'rb')
            # eval('%s = pickle.load(f)' % k)
            yield eval('pickle.load(f)')
            f.close()
