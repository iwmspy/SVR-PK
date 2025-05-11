import numpy as np
import pandas as pd
import os
import shutil
import time
import logging
from typing import Optional

def dfconcatinatorwithlabel(dfs : dict, label: str='label'):
    '''
        dfs : {label : df, ...}
    '''
    dfs_ = []
    for l, df in dfs.items():
        df_ = df.copy()
        df_[label] = l
        dfs_.append(df_)
    new_df = pd.concat(dfs_)
    return new_df

def mean_of_top_n_elements(arr, n):
    sorted_indices = np.argsort(-arr, axis=1)
    top_n_values = np.take_along_axis(arr, sorted_indices[:, :n], axis=1)
    mean_values = np.mean(top_n_values, axis=1).reshape(-1,1)
    return mean_values

def tsv_merge(tsv_list:list, merge_file:list, raw_df=False, chunksize:int=10000):
    if not(raw_df):
        first = pd.read_table(tsv_list[0],index_col=0,header=0,
                            chunksize=chunksize)
    else:
        first = [tsv_list[0]]
    # print(f'Merging {tsv_list[0]} -> {merge_file}')
    with open(merge_file,'w') as f:
        fst = True
        for _ in first:
            if fst:
                f.write('\t'+'\t'.join(_.columns.to_list()))
                f.write('\n')
                fst = False
            for idx, rows in _.iterrows():
                r = [str(idx)]
                r.extend(list(map(str, rows)))
                f.write('\t'.join(r))
                f.write('\n')
    if len(tsv_list)==1:return
    for path in tsv_list[1:]:
        # print(f'Merging {path} -> {merge_file}')
        if not(raw_df):
            df = pd.read_table(path,index_col=0,header=0,
                            chunksize=chunksize)
        else:
            df = [path]
        with open(merge_file,'a') as f:
            for _ in df:
                for idx, rows in _.iterrows():
                    r = [str(idx)]
                    r.extend(list(map(str, rows)))
                    f.write('\t'.join(r))
                    f.write('\n')
    return

class mkdir:
    def __init__(self):
        pass

    def mk_dir(self,dir_path,overwrite=False):
        # Check existing
        if os.path.isdir(dir_path):
            print(f"Directry '{dir_path}' already exist.")
            if overwrite:
                print(f"Overwrite option is selected. Directry will be overwritten.")
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
                print(f"Directry '{dir_path}' was overwritten.")
        else:
            print(f"Directry '{dir_path}' not exist.")
            os.makedirs(dir_path)
            print(f"Directry '{dir_path}' was created.")    

    def mk_dir_list(self,dirlist):
        for dir in dirlist:
            self.mk_dir(dir)

def check_group_vals(group,thres=1):
    g = group.max()-group.min()
    if g>=thres:
        return 1
    elif g==0 and len(group)==1:
        return -1
    else:
        return 0

class timer:
    def __init__(self, process_name : str = None, round : int = 3):
        self.process_name = process_name \
            if process_name is not None else 'process'
        self.round = round

    def __enter__(self):
        self.start = time.time()
        self.stmsg = f'---{self.process_name} start---'
        print(self.stmsg)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        end   = time.time()
        runt  = end - self.start
        edmsg = f'---{self.process_name} end. (Took {round(runt, self.round)} sec)---'
        print(edmsg)
    
    def get_runtime(self):
        now   = time.time()
        return round(now - self.start, self.round)


def AttrJudge(d: dict, key, default_value):
    val = d[key] if key in d else default_value
    d[key] = val
    return val

def MakeDirIfNotExisting(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def ArraySplitByN(array: np.array, n: int):
    for i in range(0, len(array), n):
        yield array[i: i+n]

class logger:
    def __init__(self, name: str = None, level: str = "INFO", filename: Optional[str] = None) -> None:
        """
        Basic logger for this repo.
        :param name: usually the file name which can be passed to the get_logger function like this get_logger(__name__)
        :param level: logging level
        :param filename: Filename to write logging to. If None, logging will print to screen.
        """
        if name is None:
            name = "RMLogger"
        self.log = logging.getLogger(name)
        self.log.setLevel(level)

        # Remove existing handlers to avoid duplicate logs
        if self.log.hasHandlers():
            self.log.handlers.clear()

        # Set up a new handler
        if filename is not None:
            MakeDirIfNotExisting(os.path.dirname(filename))
            handler = logging.FileHandler(filename)
        else:
            handler = logging.StreamHandler()

        # Set formatter and add handler
        formatter = logging.Formatter(
            fmt="%(asctime)s,%(msecs)d %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.log.addHandler(handler)
    
    def write(self, msg):
        self.log.info(msg)

if __name__=='__main__':
    print(1)