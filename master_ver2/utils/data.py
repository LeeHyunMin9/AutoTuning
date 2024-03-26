import pandas as pd
import numpy as np



# Unify the data types of the columns in the dataframe ========================================>>

def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):
    """Shrink a dataframe.

    Return any possible smaller data types for DataFrame columns.
    Allows `object`->`category`, `int`->`uint`, and exclusion.
    From: https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py

    """

    # 1: Build column filter and typemap
    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    # no int16 as orjson in plotly doesn't support it
    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }

    if obj2cat:
        # User wants to categorify dtype('Object'), which may not always save space
        typemap["object"] = "category"
    else:
        excl_types.add("object")

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t

    return df.astype(new_dtypes)