import numpy as np
import pandas as pd
import os
import shutil
import datetime
import time
import logging
from typing import Optional
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def iterable(obj):
    return hasattr(obj,"__iter__")


def select_dataset(df:pd.DataFrame, subset, 
                   col_name:str, mostselect:str=None, threshold=100):
    # Select dataset by threshold
    # If analyzed value(s) are over threshold, return col name
    df_selected = df[df[subset]>=threshold]
    if mostselect==None:
        return df_selected[col_name].to_list()
    idx = df_selected.groupby(mostselect)
    df_selected_m = df_selected.loc[idx[subset].idxmax(),:].sort_values(subset,ascending=False)
    return df_selected_m[col_name].to_list()


def mean_of_top_n_elements(arr, n):
    sorted_indices = np.argsort(-arr, axis=1)
    top_n_values = np.take_along_axis(arr, sorted_indices[:, :n], axis=1)
    mean_values = np.mean(top_n_values, axis=1).reshape(-1,1)
    return mean_values


def oldfile_mover(path:str,depth:int=0):
    '''Written by iwmspy (20231205)
        When you want to preserve an older file, you can do it by using this code.
        path  : Your file path which you want to preserve. 
                This must be following shape.
                  "(your_dir_path)/(your_file_prefix).(your_file_extension)"
        depth : Where you want to preserve an older file.
                The integer indicates the depth of nodes.
                For example, when we assume the following structure of nodes,
                  /root/dir1/dir2/dir3/file.ext
                If depth = 1, your older file will be preserved in following dir.
                  /root/dir1/dir2/past/(creation_date)
                !NOTE!
                If path is relative path, you CANNOT desinate shallower depth than current dir.
    '''
    try:
        if os.path.exists(path):
            ex = f"File '{path}' already exist."
            print(ex)

            name = path.rsplit('/',1)[-1].rsplit(".",1)
            file_stat = os.stat(path)
            creation_date = str(datetime.datetime.fromtimestamp(file_stat.st_ctime).strftime("%Y-%m-%d"))
            creation_time = str(datetime.datetime.fromtimestamp(file_stat.st_ctime).strftime("%H-%M-%S"))
            creation = f'{creation_date}_{creation_time}'

            if isinstance(depth,int) and (depth >= 0) and (len(path.split('/'))-1 > depth):
                folder_path = f"{path.rsplit('/',depth+1)[0]}/past/{creation_date}"
            else:
                raise ValueError("Your set 'depth' is invalid number.")
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            pref  = f'{name[0]}_{creation}'
            pref_ = f'{name[0]}_{creation}'
            num = 1
            
            while os.path.exists(f'{folder_path}/{pref}.{name[-1]}'):
                pref = f'{pref_}_{num}'
                num += 1

            shutil.move(path, f'{folder_path}/{pref}.{name[-1]}')

            done = f"File '{path}' was renamed and moved to -> '{folder_path}/{pref}.{name[-1]}'"
            print(done)
        else:
            nex = f"File '{path}' not exist."
            print(nex)

    except Exception as e:
        raise Exception(f'!!!Error!!! : {e}')
    
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
    

def calc_ltm(x):
    return (x*(x-1))/2


def check_dif(df, thres=1.0):
    return abs(df['pred_fr_1']-df['pred_fr_2'])>thres


def df_mask(df:pd.DataFrame,mask_val=-1):
    assert df.shape[0]==df.shape[1],\
        "If df is not a square matrix, can't mask the matrix."
    matrix = df.copy()
    matrix = matrix.to_numpy()
    matrix[np.triu_indices(n=len(matrix), k=1)] = mask_val
    return pd.DataFrame(matrix, columns=df.columns, index=df.index)


def CustomTrainTestSplit(
        data : pd.DataFrame,
        train_size : float  = 0.8,
        random_state : int  = 0,
        custom_train : dict = None,
    ):

    if custom_train != None:
        assert(isinstance(custom_train, dict) and len(custom_train)==1)
        col = list(custom_train.keys())[0]
        train_val = list(custom_train.values())[0]
        df_train = data[data[col].isin(train_val)]  if col!=None else data.loc[data.index.isin(train_val)]
        df_test  = data[~data[col].isin(train_val)] if col!=None else data.loc[~data.index.isin(train_val)]

    else:    
        data_index   = data.index
        data_columns = data.columns

        X = data.values

        X_train, X_test, X_train_idx, X_test_idx = train_test_split(
            X, data_index, train_size=train_size, 
            random_state=random_state, shuffle=True
        )

        df_train = pd.DataFrame(X_train, index=X_train_idx, columns=data_columns)
        df_test  = pd.DataFrame(X_test, index=X_test_idx, columns=data_columns)

    return df_train, df_test


def SplitXandy(data:pd.DataFrame,y_col:str,x_cols:list=None):
    x = data.loc[:, x_cols] if x_cols != None else data.loc[:,~data.columns.isin(y_col)]
    y = data[y_col]
    return x, y

def GetT(return_second=False, return_date=False):
    t = datetime.datetime.fromtimestamp(time.time())
    if return_second:
        return '{}{}{}_{}{}{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
    elif return_date:
        return '{}{}{}'.format(t.year,t.month,t.day)
    else:
        return '{}{}{}_{}{}'.format(t.year, t.month, t.day, t.hour, t.minute)


def check_group_vals(group,thres=1):
    g = group.max()-group.min()
    if g>=thres:
        return 1
    elif g==0 and len(group)==1:
        return -1
    else:
        return 0


def ret_multi_val(*args):
    return args


def csr_as_type(array, type=bool):
    return csr_matrix(array).astype(dtype=type,copy=False)


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