# std imports
from dataclasses import dataclass
from os import PathLike
from typing import Iterable, Optional, Union
import logging


# tpl imports
import numpy as np
import pandas as pd



def get_dataset(
    train : PathLike,
    test : Optional[PathLike] = None,
    test_split : float = 0.1,
    seed : int = 42,
    drop : Optional[Iterable[str]] = None,
    dropna : Optional[Iterable[str]] = None
):
    test_split = float(test_split)
    return Dataset(train, test_fpath=test, test_split=test_split, seed=seed, drop_columns=drop, dropna_columns=dropna)



@dataclass
class Dataset:
    train : pd.DataFrame
    test : pd.DataFrame

    def __init__(self, 
        train_fpath : PathLike,
        test_fpath : Optional[PathLike] = None, 
        test_split : float = 0.1,
        seed : int = 42,
        drop_columns : Optional[Iterable[str]] = None,
        dropna_columns : Optional[Iterable[str]] = None
    ):
        self.train = pd.read_csv(train_fpath)

        if test_fpath:
            self.test = pd.read_csv(test_fpath)
        else:
            self.test = self.train.sample(frac=test_split, random_state=seed)
            self.train = self.train.drop(self.test.index)

        if drop_columns:
            self.train.drop(columns=drop_columns, inplace=True)
            self.test.drop(columns=drop_columns, inplace=True)

        if dropna_columns:
            self.train.dropna(subset=dropna_columns, inplace=True)
            self.test.dropna(subset=dropna_columns, inplace=True)
        
        self.one_hot_map_ = {}


    def is_one_hot_column(self, column_name : str) -> bool:
        return column_name in self.one_hot_map_

    def get_one_hot_columns(self, column_name : str) -> Iterable[str]:
        return self.one_hot_map_[column_name]
    
    def one_hot_encode(self, columns : str):
        self.train = pd.get_dummies(self.train, columns=columns, dummy_na=True)
        self.test = pd.get_dummies(self.test, columns=columns, dummy_na=True)
        self.test = self.test.reindex(columns=self.train.columns, fill_value=0)

        for c in columns:
            new_cols = self.train.columns[self.train.columns.str.startswith(c+'_')].to_list()
            self.one_hot_map_[c] = new_cols

    def all_columns_except(self, columns : Iterable[str]) -> Iterable[str]:
        return list(set(self.train.columns) - set(columns))

