import mmap
import numpy as np
from time import time
import os
from matplotlib.image import imread

class InputStream:
    '''
    Stream lines from a single file or a list of image files.
    Depends on numpy for 64 bit compact arrays. Works on data
    larger than available memory. Images are loaded using
    maplotlib's imgread function.
    '''
    def __init__(self,source, img_mode=False, log_rate=int(10e4)):
        self.source = source
        self.img_mode = img_mode
        self.log_rate = log_rate if log_rate else int(10e4)
        self.offsets, self.num_records = self.scan_offsets()
        np.random.shuffle(self.offsets)
            
    def __iter__(self):
        pass
    
    def __len__(self):
        return self.num_records
    
    def scan_offsets(self):
        tic = time()
        tmp_offsets = []
        print('Scanning records')
        if not self.img_mode:
            with open(self.source) as f:
                mm = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ)
                # Will stop on first empty line
                i = 0
                for line in iter(mm.readline, ''):
                    tmp_offsets.append(mm.tell())
                    i += 1
                    if i % self.log_rate == 0:
                        print('%d records scanned' % i, end='\r', flush=True)
            offsets = np.asarray(tmp_offsets, dtype='uint64')
            del tmp_offsets
            print('%d total records scanned in %.2fMB and %.2fsec'
                  %(i, os.path.getsize(self.source)/(1024*1024.0), time()-tic))
            return offsets, i
        else:
            filelist = [f for f in listdir(self.source) if isfile(join(mypath, f))]
            num_records = len(filelist)
            print('%d total records scanned in %.2fMB and %.2fsec'
                % (num_records, sum(os.path.getsize(self.source+'/'+f) for f in filelist)/(1024*1024.0), time()-tic))
            return filelist, num_records