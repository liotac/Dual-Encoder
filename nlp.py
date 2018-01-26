import numpy as np
import mmap, os, random
from collections import deque
from timeit import default_timer as timer
from matplotlib.image import imread

class RecordStream:
    '''
    Generator class that reads and stream data from either newline
    delimited text files or a directory of image files. Works for
    data larger than memory using mmap for text files and generator
    lazy image loading via matplotlib's imgread function. Yields a
    single sample at a time.

    Example
    -------
    stream = RecordStream(path)
    for line_or_img in stream:
        parsing_function(line_or_img)
    '''
    def __init__(self, source, img_mode=False, log_rate=int(10e4), skip_header=None):
        '''
        Initialize the RecordStream with a path to file/directory. In case
        of text data, the initialization will scan through the whole file
        in O(n) time and get a shuffled list of lines to be yielded. In
        the case of a directory of images, file loading will be done during
        the call to the iterable.

        Parameters
        ----------
        source: path to text file or directory of images
        img_mode: True if directory of images
        log_rate: rate at which to print to stdout the initial scan progress
        skip_header: number of lines to skip at the head of the text file
        '''
        self.source = source
        self.img_mode = img_mode
        self.log_rate = log_rate if log_rate else int(10e4)
        self.skip_header = skip_header
        self.offsets, self.num_records = self._scan_records()
        np.random.shuffle(self.offsets[self.skip_header:])

    def __len__(self):
        return self.num_records
  
    def __repr__(self):
        return self.source
      
    def __iter__(self):
        if self.img_mode:
            for img in self.offsets:
                yield imread(img)
        else:
            with open(self.source, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                for line_number, offset in enumerate(self.offsets):
                    mm.seek(offset)
                    line = mm.readline()
                    if line.strip():
                        yield line.decode('ascii')

  
    def _scan_records(self):
        tic = timer()
        if self.img_mode:
            filelist = [f for f in os.listdir(self.source) if os.path.isfile(os.path.join(self.source, f))]
            num_records = len(filelist)
            print('Records: %d found, %.2fMB, %.2fsec'
              % (num_records, sum(os.path.getsize(self.source+'/'+f) for f in filelist)/1048576.0, timer()-tic))
            return filelist, num_records
        else:
            with open(self.source, 'r+b') as f:
                offs = []
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                num_records = 0
                for line in iter(mm.readline, b''): #note the b
                    offs.append(mm.tell())
                    num_records += 1
                    if num_records % self.log_rate == 0:
                        print('Records: %d scanned' % num_records, end='\r', flush=True)
                offsets = np.asarray(offs, dtype='uint64')
                del offs
                print('Records: %d scanned, %.2fMB, %.2fsec' % 
                    (num_records, os.path.getsize(self.source)/1048576.0, timer()-tic))
                return offsets[self.skip_header:], num_records
                
class ProgressTracker:
    '''
    A simple progress bar if total is given, else counts the iteration and duration
    '''
    progress_bar_width = 40
    def __init__(self, rate=1000, total=None):
        self.rate = int(rate) if rate else 1000
        self.total = int(total) if total else None
        self._counter = 0
        self._timer = timer()
        self.duration = 0
         
    def __len__(self):
        return self._counter

    def __call__(self, done=False):
        if done:
            self.duration = timer() - self._timer
            print('Progress: %d samples, %.2fsec' % (self._counter, self.duration))
        else:
            self._counter += 1
            if self._counter % self.rate == 0 or self._counter == 0:
                if self.total:
                    progress = int((float(self._counter)/self.total*self.progress_bar_width))
                    bar = progress*'='+ (self.progress_bar_width-progress)*'-' if progress == self.progress_bar_width else (progress-1)*'='+'>'+(self.progress_bar_width-progress)*'-'
                    print('Progress: %d samples, %.2fsec [' % (self._counter, timer()-self._timer) + bar + ']', end='\r', flush=True)
                else:
                    print('Progress: %d samples, %.2fsec' % (self._counter, timer()-self._timer), end='\r', flush=True)

class ContextResponse:
    '''
    Builds a training set from a set of dialogues containing utterances.
    Positively labeled sample are generated using next utterance after
    a given context size. Negative samples are generated by sampling a
    random utterance from elsewhere in the corpus.
    '''
    def __init__(self, total_dialogues, context_size, buffer_size, num_negative=1):
        assert num_negative < buffer_size
        assert buffer_size <= total_dialogues
        self.total_dialogues = int(total_dialogues)
        self.context_size = int(context_size)
        self.buffer_size = int(buffer_size)
        self.num_negative = int(num_negative) if num_negative else 1
        self.buffer = deque(maxlen=self.buffer_size)
        self.initial_samples = [] #list of dialogues that aren't returned
        self.prev_utterances = [] #update buffer with prev utterances
        self.counter = 0
    
    def __repr__(self):
        return str({'buffer': len(self.buffer),
                'context_size': self.context_size,
                'buffer_size': self.buffer_size,
                'num_negative': self.num_negative,
                'initial_samples': len(self.initial_samples)})
    
    def __call__(self, dialogue):
        '''
        Usage is to call the object on conversation, represented as a list
        of utterances (list of list), which should already have been
        tokenized (list of list of list).
        '''
        self.counter += 1
        self.buffer.extend(self.prev_utterances[:min(len(self.prev_utterances), self.buffer.maxlen-len(self.buffer))])
        self.prev_utterances = dialogue[:min((len(dialogue)-self.context_size)*self.num_negative, len(dialogue))]
        if len(self.buffer) < self.buffer_size:
            i = len(dialogue)
            while i and len(self.buffer) < self.buffer_size:
                i -= 1
                self.buffer.append(dialogue[i])
            self.initial_samples.append(dialogue)
            return []
        else:
            if self.counter == self.total_dialogues:
                dialogues = [dialogue]
                dialogues.extend(self.initial_samples)
                del self.initial_samples
                self.initial_samples = []
                self.counter = 0
                for dialogue in dialogues:
                    yield from self.__call__(dialogue)
            else:
                yield from self.create_pairs(dialogue)
                
    def create_pairs(self, dialogue):
        '''
        Creates a context-response pair from a given list of utterances in
        a dialogue. Negative responses are sampled from the buffer.
        '''
        if not dialogue or not (len(dialogue) > self.context_size):
            return []
        for i in range(len(dialogue) - self.context_size):
            context = dialogue[i:i + self.context_size]
            response = dialogue[i + self.context_size]
            
            # Pick self.num_negative negative samples
            loop_c = len(self.buffer)
            for _ in range(self.num_negative):
                self.buffer.rotate(random.randrange(self.buffer_size))
                try:
                    negative = self.buffer.pop()
                except IndexError:
                    raise IndexError('pop from an empty deque, buffer too small for the given (large) num_negative or (small) context_size')
                else:
                    while (negative in context or negative == response) and loop_c:
                        self.buffer.append(negative)
                        self.buffer.rotate(random.randrange(self.buffer_size))
                        negative = self.buffer.pop()
                        loop_c -= 1
                    if not loop_c:
                        raise Exception('Loop detected. All buffer utterances are in context or response. Consider increasing buffer size.')
                    yield {'context': context,
                           'response': negative,
                           'label': 0}
            yield {'context': context,
                   'response': response,
                   'label': 1}