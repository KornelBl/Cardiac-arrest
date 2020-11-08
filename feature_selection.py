import pandas as pd
import numpy as np
import glob
from sklearn.feature_selection import SelectKBest, chi2


def get_k_best(x, y, k: int = 10):
    fvalue_selector = SelectKBest(chi2, k=k)
    fvalue_selector.fit(x, y)
    return fvalue_selector.transform(x)


