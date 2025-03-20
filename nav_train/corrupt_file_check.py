import glob
import bz2
import tqdm
import _pickle as cPickle
import multiprocessing as mp


def load_path(path):
    try:
        with bz2.BZ2File(path, 'rb') as fp:
            data = cPickle.load(fp)
            in_semmap = data["in_semmap"]
    except:
        print(f'====> Path {path} is corrupt')


paths = glob.glob("*/precomputed_dataset_24.0_123_spath_square/*/*/*.pbz2")
print(f"# paths: {len(paths)}")


pool = mp.Pool(80, maxtasksperchild=8)
_ = list(tqdm.tqdm(pool.imap(load_path, paths), total=len(paths)))