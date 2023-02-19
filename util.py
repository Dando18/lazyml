# std imports
import copy
from typing import Any, Iterable


def without(d : dict, key : str, *vargs) -> dict:
    ''' Return a copy of dict without provided keys '''
    c = copy.copy(d)
    c.pop(key, None)
    for k in vargs:
        c.pop(k, None)
    return c


def unlistify(x : Iterable[Any]):
    ''' if x is a 1 element list then return that element. otherwise return list '''
    if len(x) == 1:
        return x[0]
    return x


def parse_columns(data : dict, dataset):
    num_column_params = sum(1 for k in data.keys() if k in ['columns', 'all-columns-except'])
    if num_column_params > 1:
        raise ValueError('Too many column parameters')

    if 'columns' in data:
        pass # do nothing in this case
    elif 'all-columns-except' in data:
        data['columns'] = dataset.all_columns_except(data.pop('all-columns-except'))
    

def expand_one_hot_columns(columns, dataset):
    new_cols = []
    for c in columns:
        if dataset.is_one_hot_column(c):
            new_cols.extend(dataset.get_one_hot_columns(c))
        else:
            new_cols.append(c)
    return new_cols

