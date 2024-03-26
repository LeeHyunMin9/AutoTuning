import re
from collections import OrderedDict, defaultdict
from inspect import signature

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def variable_return(X, y):
    """Return one or two arguments depending on which is None."""
    if y is None:
        return X
    elif X is None:
        return y
    else:
        return X, y


class TransformerWrapper(BaseEstimator, TransformerMixin):
    """Meta-estimator for transformers.

    Wrapper for all transformers in preprocess to return a pandas
    dataframe instead of a numpy array. Note that, in order to
    keep the correct column names, the underlying transformer is
    only allowed to add or remove columns, never both.
    From: https://github.com/tvdboom/ATOM/blob/master/atom/utils.py

    Parameters
    ----------
    transformer: estimator
        Transformer to wrap. Should implement a `fit` and/or `transform`
        method.

    include: list or None, default=None
        Columns to apply on the transformer. If specified, only these
        columns are used and the rest ignored. If None, all columns
        are used.

    exclude: list or None, default=None
        Columns to NOT apply on the transformer. If None, no columns
        are excluded.

    """

    def __init__(self, transformer, include=None, exclude=None):
        self.transformer = transformer
        self.include = include
        self.exclude = exclude

        self._train_only = getattr(transformer, "_train_only", False)
        self._include = self.include
        self._exclude = self.exclude or []
        self._feature_names_in = None

    @property
    def feature_names_in_(self):
        return self._feature_names_in

    def _name_cols(self, array, df):
        """Get the column names after a transformation.

        If the number of columns is unchanged, the original
        column names are returned. Else, give the column a
        default name if the column values changed.

        Parameters
        ----------
        array: np.ndarray
            Transformed dataset.

        df: pd.DataFrame
            Original dataset.

        """
        # If columns were only transformed, return og names
        if array.shape[1] == len(self._include):
            return self._include

        # If columns were added or removed
        temp_cols = []
        for i, col in enumerate(array.T, start=2):
            # equal_nan=True fails for non-numeric arrays
            mask = df.apply(
                lambda c: np.array_equal(
                    c,
                    col,
                    equal_nan=is_numeric_dtype(c)
                    and np.issubdtype(col.dtype, np.number),
                )
            )
            if any(mask) and mask[mask].index.values[0] not in temp_cols:
                temp_cols.append(mask[mask].index.values[0])
            else:
                # If the column is new, use a default name
                counter = 1
                while True:
                    n = f"feature {i + counter + df.shape[1] - len(self._include)}"
                    if (n not in df or n in self._include) and n not in temp_cols:
                        temp_cols.append(n)
                        break
                    else:
                        counter += 1

        return temp_cols

    def _reorder_cols(self, df, original_df):
        """Reorder the columns to their original order.

        This function is necessary in case only a subset of the
        columns in the dataset was used. In that case, we need
        to reorder them to their original order.

        Parameters
        ----------
        df: pd.DataFrame
            Dataset to reorder.

        original_df: pd.DataFrame
            Original dataframe (states the order).

        """
        # Check if columns returned by the transformer are already in the dataset
        for col in df:
            if col in original_df and col not in self._include:
                raise ValueError(
                    f"Column '{col}' returned by transformer {self.transformer} "
                    "already exists in the original dataset."
                )

        # Force new indices on old dataset for merge
        try:
            original_df.index = df.index
        except ValueError:  # Length mismatch
            raise IndexError(
                f"Length of values ({len(df)}) does not match length of "
                f"index ({len(original_df)}). This usually happens when "
                "transformations that drop rows aren't applied on all "
                "the columns."
            )

        # Define new column order
        # Use OrderedDict as ordered set (only keys matter)
        # We want a set to avoid duplicate column names, which can happen
        # if we have eg. COL_A and COL_A_2 encoded using OHE
        columns = OrderedDict()
        for col in original_df:
            if col in df or col not in self._include:
                columns[col] = None

            # Add all derivative columns: cols that originate from another
            # and start with its progenitor name, e.g. one-hot encoded columns
            columns.update(
                [
                    (c, None)
                    for c in df.columns
                    if c.startswith(f"{col}_") and c not in original_df
                ]
            )

        # Add remaining new columns (non-derivatives)
        columns.update([(col, None) for col in df if col not in columns])

        columns = list(columns.keys())

        # Merge the new and old datasets keeping the newest columns
        new_df = df.merge(
            right=original_df[[col for col in original_df if col in columns]],
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("", "__drop__"),
        )
        new_df = new_df.drop(new_df.filter(regex="__drop__$").columns, axis=1)

        return new_df[columns]

    def _prepare_df(self, X, out):
        """Convert to df and set correct column names and order."""
        # Convert to pandas and assign proper column names
        if not isinstance(out, pd.DataFrame):
            if hasattr(self.transformer, "get_feature_names_out"):
                columns = self.transformer.get_feature_names_out()
            elif hasattr(self.transformer, "get_feature_names"):
                # Some estimators have legacy method, e.g. category_encoders
                columns = self.transformer.get_feature_names()
            else:
                columns = self._name_cols(out, X)

            out = pd.DataFrame(out, index=X.index, columns=columns)

        # Reorder columns if only a subset was used
        if len(self._include) != X.shape[1]:
            return self._reorder_cols(out, X)
        else:
            return out

    def fit(self, X=None, y=None, **fit_params):
        # Save the incoming feature names
        self.target_name_ = None
        feature_names_in = []
        if hasattr(X, "columns"):
            feature_names_in += list(X.columns)
        if hasattr(y, "name"):
            feature_names_in += [y.name]
            self.target_name_ = y.name
        if feature_names_in:
            self._feature_names_in = feature_names_in

        args = []
        transformer_params = signature(self.transformer.fit).parameters
        if "X" in transformer_params and X is not None:
            if self._include is None:
                self._include = [
                    c for c in X.columns if c in X and c not in self._exclude
                ]
            elif not self._include:  # Don't fit if empty list
                return self
            else:
                self._include = [
                    c for c in self._include if c in X and c not in self._exclude
                ]
            args.append(X[self._include])
        if "y" in transformer_params and y is not None:
            args.append(y)

        self.transformer.fit(*args, **fit_params)
        return self

    def transform(self, X=None, y=None):
        X = pd.DataFrame(X, index=getattr(y, "index", None))
        y = pd.Series(y, index=getattr(X, "index", None), name=self.target_name_)

        args = []
        transform_params = signature(self.transformer.transform).parameters
        if "X" in transform_params:
            if X is not None:
                if self._include is None:
                    self._include = [
                        c for c in X.columns if c in X and c not in self._exclude
                    ]
                elif not self._include:  # Don't transform if empty list
                    return variable_return(X, y)
            else:
                return variable_return(X, y)
            args.append(X[self._include])
        if "y" in transform_params:
            if y is not None:
                args.append(y)
            elif "X" not in transform_params:
                return X, y

        output = self.transformer.transform(*args)

        # Transform can return X, y or both
        if isinstance(output, tuple):
            new_X = self._prepare_df(X, output[0])
            new_y = pd.Series(output[1], index=new_X.index, name=y.name)
        else:
            if len(output.shape) > 1:
                new_X = self._prepare_df(X, output)
                new_y = y if y is None else y.set_axis(new_X.index)
            else:
                new_y = pd.Series(output, index=y.index, name=y.name)
                new_X = X if X is None else X.set_index(new_y.index)

        return variable_return(new_X, new_y)


class TransformerWrapperWithInverse(TransformerWrapper):
    def inverse_transform(self, y):
        y = pd.Series(y, index=getattr(y, "index", None), name=self.target_name_)
        output = self.transformer.inverse_transform(y)
        return pd.Series(output, index=y.index, name=y.name)
    

class TargetTransformer(BaseEstimator):
    """Wrapper for a transformer to be used on target instead."""

    def __init__(self, estimator, enforce_2d: bool = True):
        self.estimator = estimator
        self._train_only = False
        self.enforce_2d = enforce_2d

    def _enforce_2d_on_y(self, y: pd.Series):
        index = y.index
        name = y.name
        if self.enforce_2d:
            if not isinstance(y, pd.DataFrame):
                y = pd.DataFrame(y, index=index, columns=[name])
        return y, index, name

    def fit(self, y: pd.Series, **fit_params):
        y, _, _ = self._enforce_2d_on_y(y)
        return self.estimator.fit(y, **fit_params)

    def transform(self, y: pd.Series):
        y, index, name = self._enforce_2d_on_y(y)
        output = self.estimator.transform(y)
        return pd.Series(output, index=index, name=name)

    def inverse_transform(self, y: pd.Series):
        y, index, name = self._enforce_2d_on_y(y)
        output = self.estimator.inverse_transform(y)
        return pd.Series(output, index=index, name=name)

    def fit_transform(self, y: pd.Series):
        y, index, name = self._enforce_2d_on_y(y)
        output = self.estimator.fit_transform(y)
        return pd.Series(output, index=index, name=name)