#!/usr/bin/env python

import atexit
import os
import re
import sys
import threading
import warnings
from argparse import ArgumentParser, ArgumentTypeError
from decimal import Decimal
from glob import glob
from itertools import product
from pprint import pprint
from shutil import rmtree
from tempfile import mkdtemp, gettempdir
from traceback import format_exception_only
from uuid import uuid4

warnings.filterwarnings('ignore', category=FutureWarning,
                        module='sklearn.utils.deprecation')
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='rpy2.robjects.pandas2ri')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.rinterface_lib.embedded as r_embedded

r_embedded.set_initoptions(
    ('rpy2', '--quiet', '--no-save', '--max-ppsize=500000'))

import rpy2.robjects as robjects
import seaborn as sns
from joblib import Memory, Parallel, delayed, dump, load, parallel_backend
from joblib._memmapping_reducer import TemporaryResourcesManager
from natsort import natsorted
from pandas.api.types import (
    is_bool_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype,
    is_object_dtype, is_string_dtype)
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.feature_selection import RFE
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer,
    RobustScaler, StandardScaler)
from sksurv.base import SurvivalAnalysisMixin
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
from tabulate import tabulate

numpy2ri.activate()
pandas2ri.activate()

from sklearn_extensions.compose import ExtendedColumnTransformer
from sklearn_extensions.feature_selection import (
    ColumnSelector, ConfidenceThreshold, EdgeRFilterByExpr, ExtendedRFE,
    MeanThreshold, MedianThreshold, NanoStringEndogenousSelector)
from sklearn_extensions.model_selection import (
    ExtendedGridSearchCV, ExtendedRandomizedSearchCV)
from sklearn_extensions.pipeline import (ExtendedPipeline,
                                         transform_feature_meta)
from sklearn_extensions.preprocessing import (
    DESeq2RLEVST, EdgeRTMMCPM, EdgeRTMMTPM, LimmaBatchEffectRemover,
    LogTransformer, NanoStringNormalizer, NanoStringDiffNormalizer)
from sklearn_extensions.utils import _determine_key_type
from sksurv_extensions.feature_selection import (
    SelectFromUnivariateSurvivalModel)
from sksurv_extensions.model_selection import (
    SurvivalStratifiedKFold, SurvivalStratifiedGroupKFold,
    SurvivalStratifiedSampleFromGroupKFold, RepeatedSurvivalStratifiedKFold,
    RepeatedSurvivalStratifiedGroupKFold,
    RepeatedSurvivalStratifiedSampleFromGroupKFold,
    SurvivalStratifiedShuffleSplit, SurvivalStratifiedGroupShuffleSplit,
    SurvivalStratifiedSampleFromGroupShuffleSplit)
from sksurv_extensions.linear_model import (
    CachedExtendedCoxnetSurvivalAnalysis, ExtendedCoxnetSurvivalAnalysis,
    ExtendedCoxPHSurvivalAnalysis, FastCoxPHSurvivalAnalysis,
    MetaCoxnetSurvivalAnalysis)
from sksurv_extensions.svm import CachedFastSurvivalSVM


def warning_format(message, category, filename, lineno, file=None, line=None):
    return ' {}: {}'.format(category.__name__, message)


def convert_to_memmap(array):
    mmap_file = os.path.join(
        joblib_temp_folder_mgr.resolve_temp_folder_name(),
        '{}-{}-{}.pkl'.format(os.getpid(), id(threading.current_thread()),
                              uuid4().hex))
    if os.path.exists(mmap_file):
        os.unlink(mmap_file)
    dump(array, mmap_file)
    return load(mmap_file, mmap_mode='r+')


def load_dataset(dataset_file):
    dataset_name, file_extension = os.path.splitext(
        os.path.split(dataset_file)[1])
    if not os.path.isfile(dataset_file) or file_extension not in (
            '.Rda', '.rda', '.RData', '.Rdata', '.Rds', '.rds'):
        raise IOError('File does not exist/invalid: {}'.format(dataset_file))
    if file_extension in ('.Rda', '.rda', '.RData', '.Rdata'):
        r_base.load(dataset_file)
        eset = robjects.globalenv[dataset_name]
    else:
        eset = r_base.readRDS(dataset_file)
    X = pd.DataFrame(r_base.t(r_biobase.exprs(eset)),
                     columns=r_biobase.featureNames(eset),
                     index=r_biobase.sampleNames(eset))
    sample_meta = r_biobase.pData(eset)
    y = Surv.from_dataframe(args.sample_meta_stat_col,
                            args.sample_meta_surv_col, sample_meta)
    if 'Group' in sample_meta.columns:
        groups = np.array(sample_meta['Group'], dtype=int)
        _, group_indices, group_counts = np.unique(
            groups, return_inverse=True, return_counts=True)
        if ('GroupWeight' in sample_meta.columns
                and sample_meta['GroupWeight'].unique().size > 1):
            group_weights = np.array(sample_meta['GroupWeight'], dtype=float)
        else:
            group_weights = None
        sample_weights = (np.max(group_counts) / group_counts)[group_indices]
    else:
        groups = None
        group_weights = None
        sample_weights = None
    try:
        feature_meta = r_biobase.fData(eset)
        feature_meta_category_cols = (
            feature_meta.select_dtypes(include='category').columns)
        feature_meta[feature_meta_category_cols] = (
            feature_meta[feature_meta_category_cols].astype(str))
    except ValueError:
        feature_meta = pd.DataFrame(index=r_biobase.featureNames(eset))
    if args.sample_meta_cols:
        new_feature_names = []
        if args.penalty_factor_meta_col in feature_meta.columns:
            raise RuntimeError('{} column already exists in feature_meta'
                               .format(args.penalty_factor_meta_col))
        feature_meta[args.penalty_factor_meta_col] = 1
        for sample_meta_col in args.sample_meta_cols:
            if sample_meta_col not in sample_meta.columns:
                raise RuntimeError('{} column does not exist in sample_meta'
                                   .format(sample_meta_col))
            if sample_meta_col in X.columns:
                raise RuntimeError('{} column already exists in X'
                                   .format(sample_meta_col))
            is_category = (is_categorical_dtype(sample_meta[sample_meta_col])
                           or is_object_dtype(sample_meta[sample_meta_col])
                           or is_string_dtype(sample_meta[sample_meta_col]))
            if args.test_dataset or not is_category:
                X[sample_meta_col] = sample_meta[sample_meta_col]
                new_feature_names.append(sample_meta_col)
            elif (args.ordinal_encode_cols is not None
                  and sample_meta_col in args.ordinal_encode_cols):
                if sample_meta_col not in ordinal_encoder_categories:
                    raise RuntimeError('No ordinal encoder categories config '
                                       'exists for {}'.format(sample_meta_col))
                if sample_meta[sample_meta_col].unique().size > 1:
                    ode = OrdinalEncoder(categories=[
                        ordinal_encoder_categories[sample_meta_col]])
                    ode.fit(sample_meta[[sample_meta_col]])
                    X[sample_meta_col] = ode.transform(
                        sample_meta[[sample_meta_col]])
                    new_feature_names.append(sample_meta_col)
            else:
                num_categories = sample_meta[sample_meta_col][
                    sample_meta[sample_meta_col] != 'NA'].unique().size
                if num_categories > 2:
                    ohe_drop = (['NA'] if 'NA' in
                                sample_meta[sample_meta_col].values else None)
                    ohe = OneHotEncoder(drop=ohe_drop, sparse=False)
                    ohe.fit(sample_meta[[sample_meta_col]])
                    new_sample_meta_cols = []
                    for category in ohe.categories_[0]:
                        if category == 'NA':
                            continue
                        new_sample_meta_col = '{}_{}'.format(
                            sample_meta_col, category).replace(' ', '_')
                        new_sample_meta_cols.append(new_sample_meta_col)
                    X = X.join(pd.DataFrame(
                        ohe.transform(sample_meta[[sample_meta_col]]),
                        index=sample_meta[[sample_meta_col]].index,
                        columns=new_sample_meta_cols), sort=False)
                    new_feature_names.extend(new_sample_meta_cols)
                elif num_categories == 2:
                    ohe = OneHotEncoder(drop='first', sparse=False)
                    ohe.fit(sample_meta[[sample_meta_col]])
                    category = ohe.categories_[0][1]
                    new_sample_meta_col = '{}_{}'.format(
                        sample_meta_col, category).replace(' ', '_')
                    X[new_sample_meta_col] = ohe.transform(
                        sample_meta[[sample_meta_col]])
                    new_feature_names.append(new_sample_meta_col)
        new_feature_meta = pd.DataFrame(index=new_feature_names)
        for feature_meta_col in feature_meta.columns:
            if (is_categorical_dtype(feature_meta[feature_meta_col])
                    or is_object_dtype(feature_meta[feature_meta_col])
                    or is_string_dtype(feature_meta[feature_meta_col])):
                new_feature_meta[feature_meta_col] = ''
            elif (is_integer_dtype(feature_meta[feature_meta_col])
                  or is_float_dtype(feature_meta[feature_meta_col])):
                new_feature_meta[feature_meta_col] = 0
            elif is_bool_dtype(feature_meta[feature_meta_col]):
                new_feature_meta[feature_meta_col] = False
        new_feature_meta[args.penalty_factor_meta_col] = (
            1 if args.penalize_sample_meta_cols else 0)
        feature_meta = feature_meta.append(new_feature_meta,
                                           verify_integrity=True)
    col_trf_col_grps = None
    if num_col_trfs > 0:
        X_ct = X.copy()
        col_trf_col_grps = []
        for n in range(1, num_col_trfs + 1):
            col_trf_cols = []
            if hasattr(args, 'col_trf_{}_patterns'.format(n)):
                for pattern in getattr(args, 'col_trf_{}_patterns'.format(n)):
                    col_trf_cols.append(
                        X_ct.columns.str.contains(pattern, regex=True))
            elif hasattr(args, 'col_trf_{}_dtypes'.format(n)):
                for dtype in getattr(args, 'col_trf_{}_dtypes'.format(n)):
                    if dtype == 'int':
                        col_trf_cols.append(
                            X_ct.dtypes.apply(is_integer_dtype).to_numpy())
                    elif dtype == 'float':
                        col_trf_cols.append(
                            X_ct.dtypes.apply(is_float_dtype).to_numpy())
                    elif dtype == 'category':
                        col_trf_cols.append(
                            X_ct.dtypes.apply(lambda d: (
                                is_bool_dtype(d) or is_categorical_dtype(d)
                                or is_object_dtype(d) or is_string_dtype(d)))
                            .to_numpy())
            X_ct = X_ct.loc[:, col_trf_cols[0]]
            col_trf_col_grps.append(col_trf_cols)
    if col_trf_col_grps and args.max_nbytes is None:
        col_trf_col_grps = convert_to_memmap(col_trf_col_grps)
    return (dataset_name, X, y, groups, group_weights, sample_weights,
            sample_meta, feature_meta, col_trf_col_grps)


def setup_pipe_and_param_grid(cmd_pipe_steps, col_trf_col_grps=None,
                              col_trf_grp_idx=0):
    pipe_steps = []
    pipe_param_routing = None
    pipe_step_names = []
    pipe_props = {'has_selector': False, 'uses_rjava': False}
    param_grid = []
    param_grid_dict = {}
    pipe_step_keys = []
    pipe_step_types = []
    for step_idx, step_keys in enumerate(cmd_pipe_steps):
        if any(k.title() == 'None' for k in step_keys):
            pipe_step_keys.append(
                [k for k in step_keys if k.title() != 'None'] + [None])
        else:
            pipe_step_keys.append(step_keys)
        if len(step_keys) > 1:
            pipe_step_names.append('|'.join(step_keys))
        else:
            pipe_step_names.append(step_keys[0])
    for pipe_step_combo in product(*pipe_step_keys):
        params = {}
        for step_idx, step_key in enumerate(pipe_step_combo):
            if step_key:
                if step_key not in pipe_config:
                    raise RuntimeError('No pipeline config exists for {}'
                                       .format(step_key))
                estimator = pipe_config[step_key]['estimator']
                if isinstance(estimator, SurvivalAnalysisMixin) or (
                        hasattr(estimator, 'estimator') and isinstance(
                            estimator.estimator, SurvivalAnalysisMixin)):
                    step_type = 'srv'
                    if hasattr(estimator, 'get_support'):
                        pipe_props['has_selector'] = True
                elif hasattr(estimator, 'get_support'):
                    step_type = 'slr'
                    pipe_props['has_selector'] = True
                elif hasattr(estimator, 'fit_transform'):
                    step_type = 'trf'
                else:
                    raise RuntimeError('Unsupported estimator type {}'
                                       .format(estimator))
                if step_idx < len(pipe_steps):
                    if step_type != pipe_step_types[step_idx]:
                        raise RuntimeError(
                            'Different step estimator types: {} {}'
                            .format(step_type, pipe_step_types[step_idx]))
                else:
                    pipe_step_types.append(step_type)
                uniq_step_name = '{}{:d}'.format(step_type, step_idx)
                if 'param_grid' in pipe_config[step_key]:
                    for param, param_values in (
                            pipe_config[step_key]['param_grid'].items()):
                        if isinstance(param_values, (list, tuple, np.ndarray)):
                            if (isinstance(param_values, (list, tuple))
                                    and param_values or param_values.size > 0):
                                uniq_step_param = '{}__{}'.format(
                                    uniq_step_name, param)
                                if len(param_values) > 1:
                                    params[uniq_step_param] = param_values
                                    if uniq_step_param not in param_grid_dict:
                                        param_grid_dict[uniq_step_param] = (
                                            param_values)
                                else:
                                    estimator.set_params(
                                        **{param: param_values[0]})
                        elif param_values is not None:
                            estimator.set_params(**{param: param_values})
                if 'param_routing' in pipe_config[step_key]:
                    if pipe_param_routing is None:
                        pipe_param_routing = {}
                    if uniq_step_name in pipe_param_routing:
                        for param in pipe_config[step_key]['param_routing']:
                            if param not in pipe_param_routing[uniq_step_name]:
                                pipe_param_routing[uniq_step_name] = param
                    else:
                        pipe_param_routing[uniq_step_name] = (
                            pipe_config[step_key]['param_routing'])
                if step_idx == len(pipe_steps):
                    if len(pipe_step_keys[step_idx]) > 1:
                        pipe_steps.append((uniq_step_name, None))
                    else:
                        pipe_steps.append((uniq_step_name, estimator))
                if len(pipe_step_keys[step_idx]) > 1:
                    params[uniq_step_name] = [estimator]
                    if uniq_step_name not in param_grid_dict:
                        param_grid_dict[uniq_step_name] = []
                    if estimator not in param_grid_dict[uniq_step_name]:
                        param_grid_dict[uniq_step_name].append(estimator)
            else:
                uniq_step_name = pipe_step_types[step_idx] + str(step_idx)
                params[uniq_step_name] = [None]
                if uniq_step_name not in param_grid_dict:
                    param_grid_dict[uniq_step_name] = []
                if None not in param_grid_dict[uniq_step_name]:
                    param_grid_dict[uniq_step_name].append(None)
        param_grid.append(params)
    pipe = ExtendedPipeline(pipe_steps, memory=memory,
                            param_routing=pipe_param_routing)
    param_grid_estimators = {}
    for param, param_values in param_grid_dict.items():
        if any(isinstance(v, BaseEstimator) for v in param_values):
            param_grid_estimators[param] = param_values
            param_grid_dict[param] = sorted(
                ['.'.join([type(v).__module__, type(v).__qualname__])
                 if isinstance(v, BaseEstimator) else v for v in param_values],
                key=lambda x: (x is None, x))
    if isinstance(pipe[0], ColumnTransformer):
        pipe = clone(pipe)
        col_trf_name, col_trf_estimator = pipe.steps[0]
        col_trf_pipe_names = []
        col_trf_transformers = []
        col_trf_param_grids = []
        col_trf_param_routing = None
        col_trf_pipe_steps = getattr(
            args, 'col_trf_{}_pipe_steps'.format(col_trf_grp_idx + 1))
        col_trf_remainder = getattr(
            args, 'col_trf_{}_remainder'.format(col_trf_grp_idx + 1))
        for trf_idx, trf_pipe_steps in enumerate(col_trf_pipe_steps):
            (trf_pipe, trf_pipe_step_names, trf_pipe_props, trf_param_grid,
             trf_param_grid_dict, trf_param_grid_estimators) = (
                 setup_pipe_and_param_grid(
                     trf_pipe_steps, col_trf_col_grps, col_trf_grp_idx + 1))
            col_trf_pipe_names.append('->'.join(trf_pipe_step_names))
            uniq_trf_name = 'trf{:d}'.format(trf_idx)
            trf_cols = col_trf_col_grps[col_trf_grp_idx][trf_idx]
            col_trf_transformers.append((uniq_trf_name, trf_pipe, trf_cols))
            if trf_param_grid:
                col_trf_param_grids.append(
                    [{'{}__{}__{}'.format(col_trf_name, uniq_trf_name, k): v
                      for k, v in params.items()}
                     for params in trf_param_grid])
                for param, param_value in trf_param_grid_dict.items():
                    param_grid_dict['{}__{}__{}'.format(
                        col_trf_name, uniq_trf_name, param)] = param_value
                for param, estimator in trf_param_grid_estimators.items():
                    param_grid_estimators['{}__{}__{}'.format(
                        col_trf_name, uniq_trf_name, param)] = clone(estimator)
            if trf_pipe.param_routing is not None:
                if col_trf_param_routing is None:
                    col_trf_param_routing = {}
                col_trf_param_routing[uniq_trf_name] = list(
                    {v for l in trf_pipe.param_routing.values() for v in l})
            for trf_pipe_prop, trf_pipe_prop_value in trf_pipe_props.items():
                if trf_pipe_prop_value:
                    pipe_props[trf_pipe_prop] = trf_pipe_prop_value
        if col_trf_param_grids:
            final_estimator_param_grid = param_grid.copy()
            param_grid = []
            for param_grid_combo in product(final_estimator_param_grid,
                                            *col_trf_param_grids):
                param_grid.append({k: v for params in param_grid_combo
                                   for k, v in params.items()})
        col_trf_estimator.set_params(param_routing=col_trf_param_routing,
                                     remainder=col_trf_remainder,
                                     transformers=col_trf_transformers)
        if col_trf_param_routing is not None:
            pipe_param_routing = (pipe.param_routing if pipe.param_routing
                                  else {})
            pipe_param_routing[col_trf_name] = list(
                {v for l in col_trf_param_routing.values() for v in l})
            pipe.set_params(param_routing=pipe_param_routing)
        pipe_step_names[0] = '||'.join(col_trf_pipe_names)
    return (pipe, pipe_step_names, pipe_props, param_grid, param_grid_dict,
            param_grid_estimators)


def col_trf_info(col_trf):
    col_trf_col_strs = []
    for trf_name, trf_transformer, trf_cols in col_trf.transformers:
        col_trf_col_strs.append('{}: {:d}'.format(
            trf_name, (np.count_nonzero(trf_cols)
                       if _determine_key_type(trf_cols) == 'bool'
                       else trf_cols.shape[0])))
        if (isinstance(trf_transformer, Pipeline)
                and isinstance(trf_transformer[0], ColumnTransformer)):
            col_trf_col_strs.append(col_trf_info(trf_transformer[0]))
    return '({})'.format(' '.join(col_trf_col_strs))


def get_param_type(param):
    pipe_step_type_regex = re.compile(
        r'^({})\d+$'.format('|'.join(pipeline_step_types)))
    param_parts = param.split('__')
    param_parts_start_idx = [i for i, p in enumerate(param_parts)
                             if pipe_step_type_regex.match(p)][-1]
    param_parts[param_parts_start_idx] = pipe_step_type_regex.sub(
        r'\1', param_parts[param_parts_start_idx])
    param_type = '__'.join(param_parts[param_parts_start_idx:])
    return param_type


def fit_pipeline(X, y, steps, params=None, param_routing=None,
                 fit_params=None, verbose=0):
    pipe = ExtendedPipeline(steps, memory=memory, param_routing=param_routing)
    if params is None:
        params = {}
    pipe.set_params(**params)
    if fit_params is None:
        fit_params = {}
    try:
        pipe.fit(X, y, **fit_params)
    except ArithmeticError as e:
        warnings.formatwarning = warning_format
        warnings.warn('Estimator fit failed. Details: {}'
                      .format(format_exception_only(type(e), e)[0]),
                      category=FitFailedWarning)
        pipe = None
    if verbose == 0:
        print('.' if pipe is not None else 'x', end='', flush=True)
    return pipe


def calculate_test_scores(estimator, X_test, y_test, metrics, y_train=None,
                          test_times=None, predict_params=None,
                          score_params=None):
    scores = {}
    if predict_params is None:
        predict_params = {}
    y_pred = estimator.predict(X_test, **predict_params)
    scores['y_pred'] = y_pred
    if score_params is None:
        score_params = {}
    if isinstance(metrics, str):
        metrics = [metrics]
    for metric in metrics:
        if metric in ('concordance_index_censored', 'score'):
            scores[metric] = concordance_index_censored(
                y_test[y_test.dtype.names[0]], y_test[y_test.dtype.names[1]],
                y_pred)[0]
        elif metric == 'concordance_index_ipcw':
            scores[metric] = concordance_index_ipcw(y_train, y_test, y_pred)[0]
        elif metric == 'cumulative_dynamic_auc':
            scores[metric] = cumulative_dynamic_auc(y_train, y_test, y_pred,
                                                    test_times)[1]
    return scores


def get_final_feature_meta(pipe, feature_meta):
    for estimator in pipe:
        feature_meta = transform_feature_meta(estimator, feature_meta)
    feature_weights = None
    final_estimator = pipe[-1]
    if hasattr(final_estimator, 'coef_'):
        feature_weights = final_estimator.coef_
    elif hasattr(final_estimator, 'feature_importances_'):
        feature_weights = final_estimator.feature_importances_
    elif hasattr(final_estimator, 'estimator_'):
        if hasattr(final_estimator.estimator_, 'coef_'):
            feature_weights = final_estimator.estimator_.coef_
        elif hasattr(final_estimator.estimator_, 'feature_importances_'):
            feature_weights = final_estimator.estimator_.feature_importances_
    if feature_weights is not None:
        if (isinstance(final_estimator, CoxnetSurvivalAnalysis)
                or (hasattr(final_estimator, 'estimator_') and isinstance(
                    final_estimator.estimator_, CoxnetSurvivalAnalysis))):
            feature_weights = np.ravel(feature_weights)
        feature_mask = feature_weights != 0
        if args.penalty_factor_meta_col in feature_meta.columns:
            feature_mask[feature_meta[args.penalty_factor_meta_col]
                         == 0] = True
        feature_meta = feature_meta.copy()
        feature_meta = feature_meta.loc[feature_mask]
        feature_meta['Weight'] = feature_weights[feature_mask]
    feature_meta.index.rename('Feature', inplace=True)
    return feature_meta


def add_param_cv_scores(search, param_grid_dict, param_cv_scores=None):
    if param_cv_scores is None:
        param_cv_scores = {}
    for param, param_values in param_grid_dict.items():
        if len(param_values) == 1:
            continue
        param_cv_values = search.cv_results_['param_{}'.format(param)]
        if any(isinstance(v, BaseEstimator) for v in param_cv_values):
            param_cv_values = np.array(
                ['.'.join([type(v).__module__, type(v).__qualname__])
                 if isinstance(v, BaseEstimator) else v
                 for v in param_cv_values])
        if param not in param_cv_scores:
            param_cv_scores[param] = {}
        for metric in args.scv_scoring:
            if metric not in param_cv_scores[param]:
                param_cv_scores[param][metric] = {'scores': [], 'stdev': []}
            param_metric_scores = param_cv_scores[param][metric]['scores']
            param_metric_stdev = param_cv_scores[param][metric]['stdev']
            if args.param_cv_score_meth == 'best':
                for param_value_idx, param_value in enumerate(param_values):
                    mean_cv_scores = (search.cv_results_
                                      ['mean_test_{}'.format(metric)]
                                      [param_cv_values == param_value])
                    std_cv_scores = (search.cv_results_
                                     ['std_test_{}'.format(metric)]
                                     [param_cv_values == param_value])
                    if mean_cv_scores.size > 0:
                        if param_value_idx < len(param_metric_scores):
                            param_metric_scores[param_value_idx] = np.append(
                                param_metric_scores[param_value_idx],
                                mean_cv_scores[np.argmax(mean_cv_scores)])
                            param_metric_stdev[param_value_idx] = np.append(
                                param_metric_stdev[param_value_idx],
                                std_cv_scores[np.argmax(mean_cv_scores)])
                        else:
                            param_metric_scores.append(np.array(
                                [mean_cv_scores[np.argmax(mean_cv_scores)]]))
                            param_metric_stdev.append(np.array(
                                [std_cv_scores[np.argmax(mean_cv_scores)]]))
                    elif param_value_idx < len(param_metric_scores):
                        param_metric_scores[param_value_idx] = np.append(
                            param_metric_scores[param_value_idx], [np.nan])
                        param_metric_stdev[param_value_idx] = np.append(
                            param_metric_stdev[param_value_idx], [np.nan])
                    else:
                        param_metric_scores.append(np.array([np.nan]))
                        param_metric_stdev.append(np.array([np.nan]))
            elif args.param_cv_score_meth == 'all':
                for param_value_idx, param_value in enumerate(param_values):
                    for split_idx in range(search.n_splits_):
                        split_cv_scores = (search.cv_results_
                                           ['split{:d}_test_{}'
                                            .format(split_idx, metric)]
                                           [param_cv_values == param_value])
                        if split_cv_scores.size > 0:
                            if param_value_idx < len(param_metric_scores):
                                param_metric_scores[param_value_idx] = (
                                    np.append(
                                        param_metric_scores[param_value_idx],
                                        split_cv_scores))
                            else:
                                param_metric_scores.append(split_cv_scores)
                        elif param_value_idx < len(param_metric_scores):
                            param_metric_scores[param_value_idx] = np.append(
                                param_metric_scores[param_value_idx], [np.nan])
                        else:
                            param_metric_scores.append([np.nan])
    return param_cv_scores


def plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                          param_cv_scores):
    metric_colors = sns.color_palette(args.sns_color_palette,
                                      len(args.scv_scoring))
    for param in param_cv_scores:
        mean_cv_scores, std_cv_scores = {}, {}
        for metric in args.scv_scoring:
            param_metric_scores = param_cv_scores[param][metric]['scores']
            param_metric_stdev = param_cv_scores[param][metric]['stdev']
            if any(len(scores) > 1 for scores in param_metric_scores):
                mean_cv_scores[metric], std_cv_scores[metric] = [], []
                for param_value_scores in param_metric_scores:
                    mean_cv_scores[metric].append(
                        np.nanmean(param_value_scores))
                    std_cv_scores[metric].append(
                        np.nanstd(param_value_scores))
            else:
                mean_cv_scores[metric] = np.ravel(param_metric_scores)
                std_cv_scores[metric] = np.ravel(param_metric_stdev)
        plt.figure(figsize=(args.fig_width, args.fig_height))
        param_type = get_param_type(param)
        if param_type in params_lin_xticks:
            x_axis = param_grid_dict[param]
            if all(0 <= x <= 1 for x in x_axis):
                if len(x_axis) <= 15:
                    plt.xticks(x_axis)
            elif len(x_axis) <= 30:
                plt.xticks(x_axis)
        elif param_type in params_log_xticks:
            x_axis = np.ravel(param_grid_dict[param])
            plt.xscale('log', base=(2 if np.all(np.frexp(x_axis)[0] == 0.5)
                                    else 10))
        elif param_type in params_fixed_xticks:
            x_axis = range(len(param_grid_dict[param]))
            xtick_labels = [v.split('.')[-1]
                            if param_type in pipeline_step_types
                            and not args.long_label_names
                            and v is not None else str(v)
                            for v in param_grid_dict[param]]
            plt.xticks(x_axis, xtick_labels)
        else:
            raise RuntimeError('No ticks config exists for {}'
                               .format(param_type))
        plt.xlim([min(x_axis), max(x_axis)])
        plt.suptitle('Effect of {} on CV Performance Metrics'.format(param),
                     fontsize=args.title_font_size)
        plt.title('{}\n{}'.format(dataset_name, pipe_name),
                  fontsize=args.title_font_size - 2)
        plt.xlabel(param, fontsize=args.axis_font_size)
        plt.ylabel('CV Score', fontsize=args.axis_font_size)
        for metric_idx, metric in enumerate(args.scv_scoring):
            plt.plot(x_axis, mean_cv_scores[metric],
                     color=metric_colors[metric_idx], lw=2, alpha=0.8,
                     label='Mean {}'.format(metric_label[metric]))
            plt.fill_between(x_axis,
                             [m - s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             [m + s for m, s in zip(mean_cv_scores[metric],
                                                    std_cv_scores[metric])],
                             alpha=0.1, color=metric_colors[metric_idx],
                             label=(r'$\pm$ 1 std. dev.'
                                    if metric_idx == len(args.scv_scoring) - 1
                                    else None))
        plt.legend(loc='lower right', fontsize='medium')
        plt.tick_params(labelsize=args.axis_font_size)
        plt.grid(True, alpha=0.3)


def get_coxnet_max_num_alphas(search):
    if args.scv_type == 'grid':
        param_combos = ParameterGrid(search.param_grid)
    elif args.scv_type == 'rand':
        param_combos = ParameterSampler(search.param_grid,
                                        n_iter=args.scv_n_iter,
                                        random_state=args.random_seed)
    max_num_alphas = 0
    pipe = search.estimator
    srv_step_name = pipe.steps[-1][0]
    cnet_srv_n_param = '{}__estimator__n_alphas'.format(srv_step_name)
    for params in param_combos:
        if (isinstance(pipe[-1], MetaCoxnetSurvivalAnalysis)
                or (srv_step_name in params and isinstance(
                    params[srv_step_name], MetaCoxnetSurvivalAnalysis))):
            max_num_alphas = max(max_num_alphas,
                                 params[cnet_srv_n_param]
                                 if cnet_srv_n_param in params
                                 else params[srv_step_name].estimator.n_alphas
                                 if srv_step_name in params
                                 else pipe[-1].estimator.n_alphas)
    return max_num_alphas


def add_coxnet_alpha_param_grid(search, X, y, pipe_fit_params):
    cnet_pipes = []
    if args.scv_type == 'grid':
        param_combos = ParameterGrid(search.param_grid)
    elif args.scv_type == 'rand':
        param_combos = ParameterSampler(search.param_grid,
                                        n_iter=args.scv_n_iter,
                                        random_state=args.random_seed)
    pipe = search.estimator
    srv_step_name = pipe.steps[-1][0]
    for params in param_combos:
        if (isinstance(pipe[-1], MetaCoxnetSurvivalAnalysis)
                or (srv_step_name in params and isinstance(
                    params[srv_step_name], MetaCoxnetSurvivalAnalysis))):
            cnet_pipe = clone(pipe)
            cnet_pipe.set_params(**params)
            cnet_pipe.steps[-1] = (srv_step_name, cnet_pipe[-1].estimator)
            for param in cnet_pipe.get_params(deep=True).keys():
                param_parts = param.split('__')
                if param_parts[-1] == 'fit_baseline_model':
                    cnet_pipe.set_params(**{param: False})
            cnet_pipes.append(cnet_pipe)
    print('Generating CoxnetSurvivalAnalysis alpha path for {} pipeline{}'
          .format(len(cnet_pipes), 's' if len(cnet_pipes) > 1 else ''),
          flush=True, end='\n' if args.scv_verbose > 0 else ' ')
    fitted_cnet_pipes = Parallel(
        n_jobs=args.n_jobs, backend=args.parallel_backend,
        max_nbytes=args.max_nbytes, verbose=args.scv_verbose)(
            delayed(fit_pipeline)(X, y, cnet_pipe.steps, params=None,
                                  param_routing=cnet_pipe.param_routing,
                                  fit_params=pipe_fit_params,
                                  verbose=args.scv_verbose)
            for cnet_pipe in cnet_pipes)
    if args.scv_verbose == 0:
        print(flush=True)
    if all(p is None for p in fitted_cnet_pipes):
        raise RuntimeError('All CoxnetSurvivalAnalysis alpha path pipelines '
                           'failed')
    param_grid = []
    cnet_pipes_idx = 0
    cnet_srv_a_param = '{}__alpha'.format(srv_step_name)
    for params in param_combos:
        param_grid.append({k: [v] for k, v in params.items()})
        if (isinstance(pipe[-1], MetaCoxnetSurvivalAnalysis)
                or (srv_step_name in params and isinstance(
                    params[srv_step_name], MetaCoxnetSurvivalAnalysis))):
            if fitted_cnet_pipes[cnet_pipes_idx] is not None:
                param_grid[-1][cnet_srv_a_param] = (
                    fitted_cnet_pipes[cnet_pipes_idx][-1].alphas_)
            else:
                del param_grid[-1]
            cnet_pipes_idx += 1
    if args.scv_type == 'grid':
        search.set_params(param_grid=param_grid)
    elif args.scv_type == 'rand':
        search.set_params(param_distributions=param_grid)
    if args.verbose > 1:
        print('Param grid:')
        pprint(param_grid)
    return search


def update_coxnet_param_grid_dict(search, param_grid_dict):
    pipe = search.estimator
    srv_step_name = pipe.steps[-1][0]
    cnet_srv_a_param = '{}__alpha'.format(srv_step_name)
    cnet_srv_l_param = '{}__estimator__l1_ratio'.format(srv_step_name)
    cnet_srv_n_param = '{}__estimator__n_alphas'.format(srv_step_name)
    if any(p in search.best_params_ for p in (cnet_srv_l_param,
                                              cnet_srv_n_param)):
        best_alpha_condition = {k: v for k, v in search.best_params_.items()
                                if k in (cnet_srv_l_param, cnet_srv_n_param)}
        param_grid_dict[cnet_srv_a_param] = list(filter(
            lambda params: all(params[k] == [v] for k, v in
                               best_alpha_condition.items()),
            search.param_grid))[0][cnet_srv_a_param]
    else:
        param_grid_dict[cnet_srv_a_param] = (
            search.param_grid[0][cnet_srv_a_param])
    return param_grid_dict


def unset_pipe_memory(pipe):
    for param, param_value in pipe.get_params(deep=True).items():
        if isinstance(param_value, Memory):
            pipe.set_params(**{param: None})
    if (isinstance(pipe[0], ColumnTransformer)
            and hasattr(pipe[0], 'transformers_')):
        for _, trf_transformer, _ in pipe[0].transformers_:
            if isinstance(trf_transformer, Pipeline):
                unset_pipe_memory(trf_transformer)
    return pipe


def run_model_selection():
    (dataset_name, X, y, groups, group_weights, sample_weights, sample_meta,
     feature_meta, col_trf_col_grps) = load_dataset(args.train_dataset)
    pipe, pipe_step_names, pipe_props, param_grid, param_grid_dict, _ = (
        setup_pipe_and_param_grid(args.pipe_steps, col_trf_col_grps))
    pipe_name = '\n'.join(pipe_step_names)
    if args.sample_meta_cols:
        pipe_has_penalty_factor = False
        for param in pipe.get_params(deep=True).keys():
            param_parts = param.split('__')
            if param_parts[-1] == 'penalty_factor_meta_col':
                pipe.set_params(**{param: args.penalty_factor_meta_col})
                pipe_has_penalty_factor = True
        for params in param_grid:
            for param_values in params.values():
                if any(isinstance(v, BaseEstimator) for v in param_values):
                    for estimator in param_values:
                        for param in estimator.get_params(deep=True).keys():
                            param_parts = param.split('__')
                            if param_parts[-1] == 'penalty_factor_meta_col':
                                estimator.set_params(
                                    **{param: args.penalty_factor_meta_col})
                                pipe_has_penalty_factor = True
        if not pipe_has_penalty_factor:
            feature_meta.drop(columns=[args.penalty_factor_meta_col],
                              inplace=True)
    if groups is not None:
        search_param_routing = {'cv': ('groups' if group_weights is None
                                       else {'groups': 'groups',
                                             'weights': 'group_weights'}),
                                'estimator': [], 'scoring': []}
    else:
        search_param_routing = None
    if pipe.param_routing:
        if search_param_routing is None:
            search_param_routing = {'estimator': [], 'scoring': []}
        for param in [p for l in pipe.param_routing.values() for p in l]:
            if param not in search_param_routing['estimator']:
                search_param_routing['estimator'].append(param)
                search_param_routing['scoring'].append(param)
    scv_scoring = None if args.scv_refit == 'score' else args.scv_scoring
    scv_refit = bool(args.test_dataset or not pipe_props['uses_rjava'])
    test_split_params = {'groups': groups} if groups is not None else {}
    pass_cv_group_weights = False
    if groups is None:
        if args.scv_use_ssplit:
            cv_splitter = SurvivalStratifiedShuffleSplit(
                n_splits=args.scv_splits, test_size=args.scv_size,
                random_state=args.random_seed)
        elif args.scv_repeats > 0:
            cv_splitter = RepeatedSurvivalStratifiedKFold(
                n_splits=args.scv_splits, n_repeats=args.scv_repeats,
                random_state=args.random_seed)
        else:
            cv_splitter = SurvivalStratifiedKFold(
                n_splits=args.scv_splits, random_state=args.random_seed,
                shuffle=True)
    elif args.scv_use_ssplit:
        if 'sample_weight' in search_param_routing['estimator']:
            cv_splitter = SurvivalStratifiedGroupShuffleSplit(
                n_splits=args.scv_splits, test_size=args.scv_size,
                random_state=args.random_seed)
        elif args.test_dataset:
            cv_splitter = SurvivalStratifiedSampleFromGroupShuffleSplit(
                n_splits=args.scv_splits, test_size=args.scv_size,
                random_state=args.random_seed)
            pass_cv_group_weights = True
        else:
            cv_splitter = SurvivalStratifiedShuffleSplit(
                n_splits=args.scv_splits, test_size=args.scv_size,
                random_state=args.random_seed)
    elif args.scv_repeats > 0:
        if 'sample_weight' in search_param_routing['estimator']:
            cv_splitter = RepeatedSurvivalStratifiedGroupKFold(
                n_splits=args.scv_splits, n_repeats=args.scv_repeats,
                random_state=args.random_seed)
        elif args.test_dataset:
            cv_splitter = RepeatedSurvivalStratifiedSampleFromGroupKFold(
                n_splits=args.scv_splits, n_repeats=args.scv_repeats,
                random_state=args.random_seed)
            pass_cv_group_weights = True
        else:
            cv_splitter = RepeatedSurvivalStratifiedKFold(
                n_splits=args.scv_splits, n_repeats=args.scv_repeats,
                random_state=args.random_seed)
    elif 'sample_weight' in search_param_routing['estimator']:
        cv_splitter = SurvivalStratifiedGroupKFold(
            n_splits=args.scv_splits, random_state=args.random_seed,
            shuffle=True)
    elif args.test_dataset:
        cv_splitter = SurvivalStratifiedSampleFromGroupKFold(
            n_splits=args.scv_splits, random_state=args.random_seed,
            shuffle=True)
        pass_cv_group_weights = True
    else:
        cv_splitter = SurvivalStratifiedKFold(
            n_splits=args.scv_splits, random_state=args.random_seed,
            shuffle=True)
    if groups is None:
        if args.test_use_ssplit:
            test_splitter = SurvivalStratifiedShuffleSplit(
                n_splits=args.test_splits, test_size=args.test_size,
                random_state=args.random_seed)
        elif args.test_repeats > 0:
            test_splitter = RepeatedSurvivalStratifiedKFold(
                n_splits=args.test_splits, n_repeats=args.test_repeats,
                random_state=args.random_seed)
        else:
            test_splitter = SurvivalStratifiedKFold(
                n_splits=args.test_splits, random_state=args.random_seed,
                shuffle=True)
    elif args.test_use_ssplit:
        if 'sample_weight' in search_param_routing['estimator']:
            test_splitter = SurvivalStratifiedGroupShuffleSplit(
                n_splits=args.test_splits, test_size=args.test_size,
                random_state=args.random_seed)
        else:
            test_splitter = SurvivalStratifiedSampleFromGroupShuffleSplit(
                n_splits=args.test_splits, test_size=args.test_size,
                random_state=args.random_seed)
            test_split_params['weights'] = group_weights
    elif args.test_repeats > 0:
        if 'sample_weight' in search_param_routing['estimator']:
            test_splitter = RepeatedSurvivalStratifiedGroupKFold(
                n_splits=args.test_splits, n_repeats=args.test_repeats,
                random_state=args.random_seed)
        else:
            test_splitter = RepeatedSurvivalStratifiedSampleFromGroupKFold(
                n_splits=args.test_splits, n_repeats=args.test_repeats,
                random_state=args.random_seed)
            test_split_params['weights'] = group_weights
    elif 'sample_weight' in search_param_routing['estimator']:
        test_splitter = SurvivalStratifiedGroupKFold(
            n_splits=args.test_splits, random_state=args.random_seed,
            shuffle=True)
    else:
        test_splitter = SurvivalStratifiedSampleFromGroupKFold(
            n_splits=args.test_splits, random_state=args.random_seed,
            shuffle=True)
        test_split_params['weights'] = group_weights
    if args.skb_slr_k_lim:
        min_train_samples = (
            X.shape[0] if args.test_dataset
            else min(train.size for train, _ in test_splitter.split(
                X, y, **test_split_params)))
        for params in param_grid:
            for param, param_values in params.items():
                param_type = get_param_type(param)
                if param_type in params_k_selected_features:
                    params[param] = param_values[
                        param_values <= min_train_samples]
        for param, param_values in param_grid_dict.items():
            param_type = get_param_type(param)
            if param_type in params_k_selected_features:
                param_grid_dict[param] = param_values[
                    param_values <= min_train_samples]
    if args.scv_type == 'grid':
        search = ExtendedGridSearchCV(
            pipe, cv=cv_splitter, error_score=args.scv_error_score,
            max_nbytes=args.max_nbytes, n_jobs=args.n_jobs,
            param_grid=param_grid, param_routing=search_param_routing,
            refit=scv_refit, return_train_score=False, scoring=scv_scoring,
            verbose=args.scv_verbose)
    elif args.scv_type == 'rand':
        search = ExtendedRandomizedSearchCV(
            pipe, cv=cv_splitter, error_score=args.scv_error_score,
            max_nbytes=args.max_nbytes, n_iter=args.scv_n_iter,
            n_jobs=args.n_jobs, param_distributions=param_grid,
            param_routing=search_param_routing, random_state=args.random_seed,
            refit=scv_refit, return_train_score=False, scoring=scv_scoring,
            verbose=args.scv_verbose)
    if args.verbose > 0:
        print(search.__repr__(N_CHAR_MAX=10000))
        if param_grid_dict:
            print('Param grid dict:')
            pprint(param_grid_dict)
    if args.verbose > 0 or args.scv_verbose > 0:
        print('Train:' if args.test_dataset else 'Dataset:', dataset_name,
              X.shape, end=' ')
        if isinstance(pipe[0], ColumnTransformer):
            print(col_trf_info(pipe[0]))
        else:
            print()
    if args.verbose > 0:
        if groups is not None:
            print('Groups:')
            pprint(groups)
            if group_weights is not None:
                print('Group weights:')
                pprint(group_weights)
        if (sample_weights is not None and 'sample_weight' in
                search_param_routing['estimator']):
            print('Sample weights:')
            pprint(sample_weights)
    pipe_fit_params = {}
    if search_param_routing:
        if 'sample_meta' in search_param_routing['estimator']:
            pipe_fit_params['sample_meta'] = sample_meta
        if 'feature_meta' in search_param_routing['estimator']:
            pipe_fit_params['feature_meta'] = feature_meta
        if 'sample_weight' in search_param_routing['estimator']:
            pipe_fit_params['sample_weight'] = sample_weights
    search_fit_params = pipe_fit_params.copy()
    if groups is not None:
        search_fit_params['groups'] = groups
        if group_weights is not None and pass_cv_group_weights:
            search_fit_params['group_weights'] = group_weights
    if args.save_model_code is not None:
        if dataset_name.split('_')[-1] == 'eset':
            model_name = '_'.join([dataset_name.rpartition('_')[0],
                                   args.save_model_code])
        else:
            model_name = '_'.join([dataset_name, args.save_model_code])
    else:
        model_name = dataset_name
    if args.load_only:
        sys.exit()
    # train w/ independent test sets
    if args.test_dataset:
        if 'CoxnetSurvivalAnalysis' in args.pipe_steps[-1]:
            search = add_coxnet_alpha_param_grid(search, X, y, pipe_fit_params)
        with parallel_backend(args.parallel_backend,
                              inner_max_num_threads=inner_max_num_threads):
            search.fit(X, y, **search_fit_params)
        if 'CoxnetSurvivalAnalysis' in args.pipe_steps[-1]:
            param_grid_dict = update_coxnet_param_grid_dict(search,
                                                            param_grid_dict)
        best_pipe = search.best_estimator_
        param_cv_scores = add_param_cv_scores(search, param_grid_dict)
        final_feature_meta = get_final_feature_meta(best_pipe, feature_meta)
        if args.verbose > 0:
            print('Model:', model_name, end=' ')
            for metric in args.scv_scoring:
                print(' {} (CV): {:.4f}'.format(
                    metric_label[metric], search.cv_results_[
                        'mean_test_{}'.format(metric)][search.best_index_]),
                      end=' ')
            print(' Params:', {
                k: ('.'.join([type(v).__module__, type(v).__qualname__])
                    if isinstance(v, BaseEstimator) else v)
                for k, v in search.best_params_.items()}, end=' ')
            if (args.penalty_factor_meta_col
                    in final_feature_meta.columns):
                num_features = final_feature_meta.loc[
                    final_feature_meta[args.penalty_factor_meta_col]
                    != 0].shape[0]
            else:
                num_features = final_feature_meta.shape[0]
            if pipe_props['has_selector'] or num_features < X.shape[1]:
                print(' Features: {:.0f}'.format(num_features))
            else:
                print()
            if 'Weight' in final_feature_meta.columns:
                print(tabulate(final_feature_meta.iloc[
                    (-final_feature_meta['Weight'].abs()).argsort()],
                               floatfmt='.6e', headers='keys'))
            else:
                print(tabulate(final_feature_meta, headers='keys'))
        plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                              param_cv_scores)
        test_datasets = natsorted(
            list(set(args.test_dataset) - set(args.train_dataset)))
        for test_dataset in test_datasets:
            (test_dataset_name, X_test, y_test, _, _, test_sample_weights,
             test_sample_meta, test_feature_meta, _) = (
                 load_dataset(test_dataset))
            pipe_predict_params = {}
            if 'sample_meta' in pipe_fit_params:
                pipe_predict_params['sample_meta'] = test_sample_meta
            if 'feature_meta' in pipe_fit_params:
                pipe_predict_params['feature_meta'] = test_feature_meta
            score_params = {'sample_weight': test_sample_weights}
            test_scores = calculate_test_scores(
                search, X_test, y_test, args.scv_scoring,
                predict_params=pipe_predict_params, score_params=score_params)
            if args.verbose > 0:
                print('Test:', test_dataset_name, end=' ')
                for metric in args.scv_scoring:
                    print(' {}: {:.4f}'.format(
                        metric_label[metric], test_scores[metric]), end=' ')
                    if metric == 'average_precision':
                        print(' PR AUC: {:.4f}'.format(test_scores['pr_auc']),
                              end=' ')
                print()
        # plot top-ranked selected features vs test performance metrics
        if 'Weight' in final_feature_meta.columns:
            fig_slr, ax_slr = plt.subplots(figsize=(args.fig_width,
                                                    args.fig_height))
            fig_slr.suptitle('Effect of Number of Top-Ranked Selected '
                             'Features on Test Performance Metrics',
                             fontsize=args.title_font_size)
            ax_slr.set_title('{}\n{}'.format(dataset_name, pipe_name),
                             fontsize=args.title_font_size - 2)
            ax_slr.set_xlabel('Number of top-ranked features selected',
                              fontsize=args.axis_font_size)
            ax_slr.set_ylabel('Test Score', fontsize=args.axis_font_size)
            tf_pipe_steps = best_pipe.steps[:-1]
            tf_pipe_steps.append(('slrc', ColumnSelector()))
            tf_pipe_param_routing = (best_pipe.param_routing
                                     if best_pipe.param_routing else {})
            tf_pipe_param_routing['slrc'] = (
                pipe_config['ColumnSelector']['param_routing'])
            if isinstance(best_pipe[-1],
                          (RFE, SelectFromUnivariateSurvivalModel)):
                final_step_name = best_pipe.steps[-1][0]
                final_estimator = best_pipe.steps[-1][1].estimator
                final_estimator_key = type(final_estimator).__qualname__
                tf_pipe_steps.append((final_step_name, final_estimator))
                if 'param_routing' in pipe_config[final_estimator_key]:
                    tf_pipe_param_routing[final_step_name] = (
                        pipe_config[final_estimator_key]['param_routing'])
                else:
                    del tf_pipe_param_routing[final_step_name]
            else:
                tf_pipe_steps.append(best_pipe.steps[-1])
            tf_pipe_fit_params = pipe_fit_params.copy()
            if 'feature_meta' not in pipe_fit_params:
                tf_pipe_fit_params['feature_meta'] = feature_meta
            tf_name_sets = []
            if args.penalty_factor_meta_col in final_feature_meta.columns:
                unpenalized_feature_names = final_feature_meta.loc[
                    final_feature_meta[args.penalty_factor_meta_col] == 0
                ].index.to_list()
                penalized_final_feature_meta = final_feature_meta.loc[
                    final_feature_meta[args.penalty_factor_meta_col] != 0]
                for feature_name in penalized_final_feature_meta.iloc[
                        (-penalized_final_feature_meta['Weight'].abs())
                        .argsort()].index:
                    tf_name_sets.append(tf_name_sets[-1] + [feature_name]
                                        if tf_name_sets else [feature_name])
                tf_name_sets = [feature_names + unpenalized_feature_names
                                for feature_names in tf_name_sets]
                x_axis = range(1, penalized_final_feature_meta.shape[0] + 1)
            else:
                for feature_name in final_feature_meta.iloc[
                        (-final_feature_meta['Weight'].abs())
                        .argsort()].index:
                    tf_name_sets.append(tf_name_sets[-1] + [feature_name]
                                        if tf_name_sets else [feature_name])
                x_axis = range(1, final_feature_meta.shape[0] + 1)
            ax_slr.set_xlim([min(x_axis), max(x_axis)])
            if len(x_axis) <= 30:
                ax_slr.set_xticks(x_axis)
            tf_pipes = Parallel(
                n_jobs=args.n_jobs, backend=args.parallel_backend,
                max_nbytes=args.max_nbytes, verbose=args.scv_verbose)(
                    delayed(fit_pipeline)(X, y, tf_pipe_steps,
                                          params={'slrc__cols': feature_names},
                                          param_routing=tf_pipe_param_routing,
                                          fit_params=tf_pipe_fit_params,
                                          verbose=args.scv_verbose)
                    for feature_names in tf_name_sets)
            if args.scv_verbose == 0:
                print(flush=True)
            test_metric_colors = sns.color_palette(
                args.sns_color_palette,
                len(test_datasets) * len(args.scv_scoring))
            for test_idx, test_dataset in enumerate(test_datasets):
                (test_dataset_name, X_test, y_test, _, _, test_sample_weights,
                 test_sample_meta, test_feature_meta, _) = (
                     load_dataset(test_dataset))
                pipe_predict_params = {}
                if 'sample_meta' in pipe_fit_params:
                    pipe_predict_params['sample_meta'] = test_sample_meta
                if 'feature_meta' in pipe_fit_params:
                    pipe_predict_params['feature_meta'] = test_feature_meta
                score_params = {'sample_weight': test_sample_weights}
                tf_test_scores = {}
                for tf_pipe in tf_pipes:
                    test_scores = calculate_test_scores(
                        tf_pipe, X_test, y_test, args.scv_scoring,
                        predict_params=pipe_predict_params,
                        score_params=score_params)
                    for metric in args.scv_scoring:
                        if metric in test_scores:
                            if metric not in tf_test_scores:
                                tf_test_scores[metric] = []
                            tf_test_scores[metric].append(test_scores[metric])
                for metric_idx, metric in enumerate(tf_test_scores):
                    ax_slr.plot(x_axis, tf_test_scores[metric], alpha=0.8,
                                lw=2, color=test_metric_colors[test_idx
                                                               + metric_idx],
                                label='{} {}'.format(test_dataset_name,
                                                     metric_label[metric]))
                ax_slr.legend(loc='lower right', fontsize='medium')
                ax_slr.tick_params(labelsize=args.axis_font_size)
                ax_slr.grid(True, alpha=0.3)
        if args.save_models:
            if args.pipe_memory:
                best_pipe = unset_pipe_memory(best_pipe)
            dump(best_pipe, '{}/{}_model.pkl'.format(args.out_dir, model_name))
    # train-test nested cv
    else:
        if args.verbose > 0:
            print('Test CV:', end=' ')
            pprint(test_splitter)
        split_models = []
        split_results = []
        param_cv_scores = {}
        param_grid_dict_alphas = None
        risk_scores = None
        base_search = clone(search)
        if 'CoxnetSurvivalAnalysis' in args.pipe_steps[-1]:
            srv_step_name = pipe.steps[-1][0]
            cnet_srv_a_param = '{}__alpha'.format(srv_step_name)
            cnet_srv_max_num_alphas = get_coxnet_max_num_alphas(base_search)
        for split_idx, (train_idxs, test_idxs) in enumerate(
                test_splitter.split(X, y, **test_split_params)):
            split_pipe_fit_params = {
                k: (v.iloc[train_idxs] if k in ('sample_meta')
                    else v[train_idxs] if k in ('sample_weight')
                    else v)
                for k, v in pipe_fit_params.items() if v is not None}
            split_search_fit_params = split_pipe_fit_params.copy()
            if groups is not None:
                split_search_fit_params['groups'] = groups[train_idxs]
                if group_weights is not None and pass_cv_group_weights:
                    split_search_fit_params['group_weights'] = (
                        group_weights[train_idxs])
            try:
                if 'CoxnetSurvivalAnalysis' in args.pipe_steps[-1]:
                    search = add_coxnet_alpha_param_grid(
                        clone(base_search), X.iloc[train_idxs], y[train_idxs],
                        split_pipe_fit_params)
                else:
                    search = clone(base_search)
                with parallel_backend(
                        args.parallel_backend,
                        inner_max_num_threads=inner_max_num_threads):
                    search.fit(X.iloc[train_idxs], y[train_idxs],
                               **split_search_fit_params)
                if pipe_props['uses_rjava']:
                    best_index = np.argmin(search.cv_results_[
                        'rank_test_{}'.format(args.scv_refit)])
                    best_params = search.cv_results_['params'][best_index]
                    best_pipe = Parallel(
                        n_jobs=args.n_jobs, backend=args.parallel_backend,
                        max_nbytes=args.max_nbytes, verbose=args.scv_verbose)(
                            delayed(fit_pipeline)(
                                X.iloc[train_idxs], y[train_idxs], pipe.steps,
                                params=pipe_params,
                                param_routing=pipe.param_routing,
                                fit_params=split_pipe_fit_params,
                                verbose=args.scv_verbose)
                            for pipe_params in [best_params])[0]
                    if args.scv_verbose == 0:
                        print(flush=True)
                else:
                    best_index = search.best_index_
                    best_params = search.best_params_
                    best_pipe = search.best_estimator_
                split_scores = {'cv': {}}
                for metric in args.scv_scoring:
                    split_scores['cv'][metric] = search.cv_results_[
                        'mean_test_{}'.format(metric)][best_index]
                split_pipe_predict_params = {
                    k: v.iloc[test_idxs] if k in ('sample_meta') else v
                    for k, v in pipe_fit_params.items()
                    if k not in ('sample_weight') and v is not None}
                split_score_params = {
                    'sample_weight': (sample_weights[test_idxs]
                                      if sample_weights is not None else None)}
                split_scores['te'] = calculate_test_scores(
                    best_pipe, X.iloc[test_idxs], y[test_idxs],
                    args.scv_scoring, predict_params=split_pipe_predict_params,
                    score_params=split_score_params)
                surv_funcs = best_pipe.predict_survival_function(
                    X.iloc[test_idxs], **split_pipe_predict_params)
            except Exception as e:
                if args.scv_error_score == 'raise':
                    raise
                if args.verbose > 0:
                    print('Model:', model_name, ' Split: {:>{width}d}'
                          .format(split_idx + 1,
                                  width=len(str(args.test_splits))), end=' ',
                          flush=True)
                warnings.formatwarning = warning_format
                warnings.warn('Estimator fit/scoring failed. This outer CV '
                              'train-test split will be ignored. Details: {}'
                              .format(format_exception_only(type(e), e)[0]),
                              category=FitFailedWarning)
                best_pipe = None
                split_result = None
            else:
                if 'CoxnetSurvivalAnalysis' in args.pipe_steps[-1]:
                    param_grid_dict = update_coxnet_param_grid_dict(
                        search, param_grid_dict)
                    full_alpha_path = np.logspace(
                        np.log10(param_grid_dict[cnet_srv_a_param][0]),
                        np.log10(param_grid_dict[cnet_srv_a_param][-1]),
                        num=cnet_srv_max_num_alphas)
                    param_grid_dict_alphas = (
                        np.vstack((param_grid_dict_alphas, full_alpha_path))
                        if param_grid_dict_alphas is not None else
                        full_alpha_path)
                param_cv_scores = add_param_cv_scores(search, param_grid_dict,
                                                      param_cv_scores)
                final_feature_meta = get_final_feature_meta(best_pipe,
                                                            feature_meta)
                if args.verbose > 0:
                    print('Model:', model_name, ' Split: {:>{width}d}'
                          .format(split_idx + 1,
                                  width=len(str(args.test_splits))), end=' ')
                    for metric in args.scv_scoring:
                        print(' {} (CV / Test): {:.4f} / {:.4f}'.format(
                            metric_label[metric], split_scores['cv'][metric],
                            split_scores['te'][metric]), end=' ')
                        if metric == 'average_precision':
                            print(' PR AUC Test: {:.4f}'.format(
                                split_scores['te']['pr_auc']), end=' ')
                    print(' Params:', {
                        k: ('.'.join([type(v).__module__,
                                      type(v).__qualname__])
                            if isinstance(v, BaseEstimator) else v)
                        for k, v in best_params.items()}, end=' ')
                    if (args.penalty_factor_meta_col
                            in final_feature_meta.columns):
                        num_features = final_feature_meta.loc[
                            final_feature_meta[args.penalty_factor_meta_col]
                            != 0].shape[0]
                    else:
                        num_features = final_feature_meta.shape[0]
                    if pipe_props['has_selector'] or num_features < X.shape[1]:
                        print(' Features: {:.0f}'.format(num_features))
                    else:
                        print()
                if args.verbose > 1:
                    if 'Weight' in final_feature_meta.columns:
                        print(tabulate(final_feature_meta.iloc[
                            (-final_feature_meta['Weight'].abs()).argsort()],
                                       floatfmt='.6e', headers='keys'))
                    else:
                        print(tabulate(final_feature_meta, headers='keys'))
                split_result = {'feature_meta': final_feature_meta,
                                'scores': split_scores,
                                'surv_funcs': surv_funcs}
                split_risk_score_col = 'Risk Score {:d}'.format(split_idx + 1)
                split_risk_scores = pd.DataFrame(
                    {split_risk_score_col: split_scores['te']['y_pred']},
                    index=sample_meta.index[test_idxs])
                risk_scores = (risk_scores.join(split_risk_scores, how='outer')
                               if risk_scores is not None else
                               split_risk_scores)
            split_results.append(split_result)
            if args.save_models:
                if args.pipe_memory and best_pipe is not None:
                    best_pipe = unset_pipe_memory(best_pipe)
                split_models.append(best_pipe)
            if args.pipe_memory:
                memory.clear(warn=False)
        if args.save_results:
            dump(split_results, '{}/{}_split_results.pkl'
                 .format(args.out_dir, model_name))
            dump(param_cv_scores, '{}/{}_param_cv_scores.pkl'
                 .format(args.out_dir, model_name))
            dump(risk_scores, '{}/{}_risk_scores.pkl'
                 .format(args.out_dir, model_name))
            r_base.saveRDS(risk_scores, '{}/{}_risk_scores.rds'
                           .format(args.out_dir, model_name))
        if args.save_models:
            dump(split_models, '{}/{}_split_models.pkl'
                 .format(args.out_dir, model_name))
        if param_grid_dict_alphas is not None:
            param_grid_dict[cnet_srv_a_param] = np.mean(
                param_grid_dict_alphas, axis=0)
        scores = {'cv': {}, 'te': {}}
        num_features = []
        for split_result in split_results:
            if split_result is None:
                continue
            for metric in args.scv_scoring:
                if metric not in scores['cv']:
                    scores['cv'][metric] = []
                    scores['te'][metric] = []
                scores['cv'][metric].append(
                    split_result['scores']['cv'][metric])
                scores['te'][metric].append(
                    split_result['scores']['te'][metric])
                if metric == 'average_precision':
                    if 'pr_auc' not in scores['te']:
                        scores['te']['pr_auc'] = []
                    scores['te']['pr_auc'].append(
                        split_result['scores']['te']['pr_auc'])
            split_feature_meta = split_result['feature_meta']
            if args.penalty_factor_meta_col in split_feature_meta.columns:
                num_features.append(split_feature_meta.loc[
                    split_feature_meta[args.penalty_factor_meta_col]
                    != 0].shape[0])
            else:
                num_features.append(split_feature_meta.shape[0])
        print('Model:', model_name, end=' ')
        for metric in args.scv_scoring:
            print(' Mean {} (CV / Test): {:.4f} / {:.4f}'.format(
                metric_label[metric], np.mean(scores['cv'][metric]),
                np.mean(scores['te'][metric])), end=' ')
            if metric == 'average_precision':
                print(' Mean PR AUC Test: {:.4f}'.format(
                    np.mean(scores['te']['pr_auc'])), end=' ')
        if num_features and (pipe_props['has_selector']
                             or np.mean(num_features) < X.shape[1]):
            print(' Mean Features: {:.0f}'.format(np.mean(num_features)))
        else:
            print()
        # feature mean rankings and scores
        feature_annots = None
        feature_weights = None
        feature_scores = {}
        for split_idx, split_result in enumerate(split_results):
            if split_result is None:
                continue
            split_feature_meta = split_result['feature_meta']
            if feature_meta.columns.any():
                if feature_annots is None:
                    feature_annots = split_feature_meta[feature_meta.columns]
                else:
                    feature_annots = pd.concat(
                        [feature_annots,
                         split_feature_meta[feature_meta.columns]], axis=0)
            elif feature_annots is None:
                feature_annots = pd.DataFrame(index=split_feature_meta.index)
            else:
                feature_annots = pd.concat(
                    [feature_annots,
                     pd.DataFrame(index=split_feature_meta.index)], axis=0)
            if 'Weight' in split_feature_meta.columns:
                if feature_weights is None:
                    feature_weights = split_feature_meta[['Weight']].copy()
                else:
                    feature_weights = feature_weights.join(
                        split_feature_meta[['Weight']], how='outer')
                feature_weights.rename(
                    columns={'Weight': 'Weight {:d}'.format(split_idx + 1)},
                    inplace=True)
            for metric in args.scv_scoring:
                if metric not in feature_scores:
                    feature_scores[metric] = pd.DataFrame(
                        split_result['scores']['te'][metric], columns=[metric],
                        index=split_feature_meta.index)
                else:
                    feature_scores[metric] = feature_scores[metric].join(
                        pd.DataFrame(split_result['scores']['te'][metric],
                                     columns=[metric],
                                     index=split_feature_meta.index),
                        how='outer')
                feature_scores[metric].rename(columns={metric: split_idx},
                                              inplace=True)
        feature_annots = feature_annots.loc[
            ~feature_annots.index.duplicated(keep='first')]
        feature_frequency = None
        feature_results = None
        feature_results_floatfmt = ['']
        if feature_weights is not None:
            feature_ranks = feature_weights.abs().rank(
                ascending=False, method='min', na_option='keep')
            if args.feature_rank_meth == 'num_select_plus1':
                feature_ranks.fillna(feature_ranks.count(axis=0) + 1,
                                     inplace=True)
            elif args.feature_rank_meth == 'num_total':
                feature_ranks.fillna(feature_ranks.shape[0], inplace=True)
            feature_frequency = feature_weights.count(axis=1)
            feature_weights.fillna(0, inplace=True)
            feature_results = feature_annots.reindex(index=feature_ranks.index,
                                                     fill_value='')
            for feature_annot_col in feature_annots.columns:
                if is_integer_dtype(feature_annots[feature_annot_col]):
                    feature_results_floatfmt.append('.0f')
                elif is_float_dtype(feature_annots[feature_annot_col]):
                    feature_results_floatfmt.append('.{:d}f'.format(
                        max(abs(Decimal(f).as_tuple().exponent)
                            for f in (feature_annots[feature_annot_col]
                                      .astype(str)))))
                else:
                    feature_results_floatfmt.append('')
            feature_results['Frequency'] = feature_frequency
            feature_results['Mean Weight Rank'] = feature_ranks.mean(axis=1)
            feature_results['Mean Weight'] = feature_weights.mean(axis=1)
            feature_results_floatfmt.extend(['.0f', '.1f', '.6e'])
        for metric in args.scv_scoring:
            if metric not in ('score', 'concordance_index_censored',
                              'concordance_index_ipcw',
                              'cumulative_dynamic_auc'):
                raise RuntimeError('No feature scores fillna value defined '
                                   'for {}'.format(metric))
            if feature_results is None:
                feature_results = feature_annots.reindex(
                    index=feature_scores[metric].index, fill_value='')
                for feature_annot_col in feature_annots.columns:
                    if is_integer_dtype(feature_annots[feature_annot_col]):
                        feature_results_floatfmt.append('.0f')
                    elif is_float_dtype(feature_annots[feature_annot_col]):
                        feature_results_floatfmt.append('.{:d}f'.format(
                            max(abs(Decimal(f).as_tuple().exponent)
                                for f in (feature_annots[feature_annot_col]
                                          .astype(str)))))
                    else:
                        feature_results_floatfmt.append('')
                feature_frequency = feature_scores[metric].count(axis=1)
                feature_results['Frequency'] = feature_frequency
                feature_results_floatfmt.append('.0f')
            feature_scores[metric].fillna(0.5, inplace=True)
            if feature_scores[metric].mean(axis=1).nunique() > 1:
                feature_results = feature_results.join(
                    pd.DataFrame({
                        'Mean Test {}'.format(metric_label[metric]):
                            feature_scores[metric].mean(axis=1)}),
                    how='left')
                feature_results_floatfmt.append('.4f')
        if args.save_results:
            dump(feature_results, '{}/{}_feature_results.pkl'
                 .format(args.out_dir, model_name))
            r_base.saveRDS(feature_results, '{}/{}_feature_results.rds'
                           .format(args.out_dir, model_name))
            if feature_weights is not None:
                dump(feature_weights, '{}/{}_feature_weights.pkl'
                     .format(args.out_dir, model_name))
                r_base.saveRDS(feature_weights, '{}/{}_feature_weights.rds'
                               .format(args.out_dir, model_name))
        if args.verbose > 0:
            print('Overall Feature Ranking:')
            if feature_weights is not None:
                print(tabulate(
                    feature_results.sort_values(by='Mean Weight Rank'),
                    floatfmt=feature_results_floatfmt, headers='keys'))
            else:
                print(tabulate(
                    feature_results.sort_values(by='Mean Test {}'.format(
                        metric_label[args.scv_refit]), ascending=False),
                    floatfmt=feature_results_floatfmt, headers='keys'))
        plot_param_cv_metrics(dataset_name, pipe_name, param_grid_dict,
                              param_cv_scores)


def run_cleanup():
    if args.pipe_memory:
        rmtree(cachedir)
    if args.parallel_backend == 'loky':
        for rtmp_dir in glob('{}/Rtmp*/'.format(args.tmp_dir)):
            rmtree(rtmp_dir)
    else:
        rmtree(r_base.tempdir()[0])


def int_list(arg):
    return list(map(int, arg.split(',')))


def str_list(arg):
    return list(map(str, arg.split(',')))


def str_bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentTypeError('Boolean value expected.')


def dir_path(path):
    if os.path.isdir(path):
        return path
    raise ArgumentTypeError('{} is not a valid path'.format(path))


def add_col_trf_args(args, argv, level=1):
    parser = ArgumentParser()
    parser.add_argument('--col-trf-{}-pipe-steps'.format(level), type=str_list,
                        nargs='+', required=True, action='append',
                        help=('ColumnTransformer {} pipeline step names'
                              .format(level)))
    parser.add_argument('--col-trf-{}-patterns'.format(level), type=str,
                        nargs='+',
                        help=('ColumnTransformer {} column patterns'
                              .format(level)))
    parser.add_argument('--col-trf-{}-dtypes'.format(level), type=str,
                        nargs='+', choices=['category', 'float', 'int'],
                        help=('ColumnTransformer {} column dtypes'
                              .format(level)))
    parser.add_argument('--col-trf-{}-remainder'.format(level), type=str,
                        choices=['drop', 'passthrough'], default='passthrough',
                        help='ColumnTransformer {} remainder'.format(level))
    args, argv = parser.parse_known_args(argv, namespace=args)
    if not (hasattr(args, 'col_trf_{}_patterns'.format(level))
            or hasattr(args, 'col_trf_{}_dtypes'.format(level))):
        parser.error('one of the following arguments is required: {}'.format(
            ' '.join(['col_trf_{}_patterns'.format(level),
                      'col_trf_{}_dtypes'.format(level)])))
    if (getattr(args, 'col_trf_{}_pipe_steps'.format(level))[0][0]
            == ['ColumnTransformer']):
        args, argv, level = add_col_trf_args(args, argv, level + 1)
    return args, argv, level


parser = ArgumentParser()
parser.add_argument('--pipe-steps', type=str_list, nargs='+', required=True,
                    help='Pipeline step names')
parser.add_argument('--train-dataset', '--dataset', '--train-eset', '--train',
                    type=str, required=True, help='training dataset')
parser.add_argument('--test-dataset', '--test-eset', '--test', type=str,
                    nargs='+', help='test datasets')
parser.add_argument('--sample-meta-stat-col', type=str, default='Status',
                    help='sample metadata survival status column name')
parser.add_argument('--sample-meta-surv-col', type=str,
                    default='Survival_in_days',
                    help='sample metadata survival days column name')
parser.add_argument('--sample-meta-cols', type=str, nargs='+',
                    help='sample metadata columns to include')
parser.add_argument('--ordinal-encode-cols', type=str, nargs='+',
                    help='sample metadata columns to ordinal encode')
parser.add_argument('--penalty-factor-meta-col', type=str,
                    default='Penalty Factor',
                    help='penalty_factor feature metadata column name')
parser.add_argument('--penalize-sample-meta-cols', default=False,
                    action='store_true',
                    help='Sample meta cols penalty factor = 1 instead of 0')
parser.add_argument('--col-slr-cols', type=str_list, nargs='+',
                    help='ColumnSelector feature or metadata columns')
parser.add_argument('--col-slr-file', type=str, nargs='+',
                    help='ColumnSelector feature or metadata columns file')
parser.add_argument('--col-slr-meta-col', type=str,
                    help='ColumnSelector feature metadata column name')
parser.add_argument('--cft-slr-thres', type=float, nargs='+',
                    help='ConfidenceThreshold threshold')
parser.add_argument('--cft-slr-meta-col', type=str,
                    default='Confidence Score',
                    help='ConfidenceThreshold feature metadata column name')
parser.add_argument('--mnt-slr-thres', type=float, nargs='+',
                    help='MeanThreshold threshold')
parser.add_argument('--mdt-slr-thres', type=float, nargs='+',
                    help='MedianThreshold threshold')
parser.add_argument('--skb-slr-k', type=int, nargs='+',
                    help='Selector k features')
parser.add_argument('--skb-slr-k-min', type=int, default=1,
                    help='Selector k min features')
parser.add_argument('--skb-slr-k-max', type=int,
                    help='Selector k max features')
parser.add_argument('--skb-slr-k-step', type=int, default=1,
                    help='Selector k step features')
parser.add_argument('--skb-slr-k-lim', default=False, action='store_true',
                    help='Selector k sample limit')
parser.add_argument('--rna-slr-mb', type=str_bool, nargs='+',
                    help='RNA slr model batch')
parser.add_argument('--ohe-trf-categories', type=str_list, nargs='+',
                    help='OneHotEncoder categories')
parser.add_argument('--ohe-trf-drop', type=str, choices=['first'],
                    help='OneHotEncoder drop')
parser.add_argument('--ord-trf-categories', type=str_list, nargs='+',
                    help='OrdinalEncoder categories')
parser.add_argument('--mms-trf-feature-range', type=int_list, default=(0, 1),
                    help='MinMaxScaler feature range')
parser.add_argument('--log-trf-base', type=int, default=2,
                    help='LogTransformer base')
parser.add_argument('--log-trf-shift', type=float, nargs='+',
                    help='LogTransformer shift')
parser.add_argument('--pwr-trf-meth', type=str, nargs='+',
                    choices=['box-cox', 'yeo-johnson'],
                    help='PowerTransformer meth')
parser.add_argument('--rna-trf-ft', type=str, nargs='+',
                    help='RNA trf fit type')
parser.add_argument('--rna-trf-mb', type=str_bool, nargs='+',
                    help='RNA trf model batch')
parser.add_argument('--rna-trf-pc', type=float, nargs='+',
                    help='RNA trf prior count')
parser.add_argument('--nsn-trf-cc', type=str, nargs='+',
                    help='NanoStringNormalizer code_count')
parser.add_argument('--nsn-trf-bg', type=str, nargs='+',
                    help='NanoStringNormalizer background')
parser.add_argument('--nsn-trf-bg-t', type=str_bool, nargs='+',
                    help='NanoStringNormalizer background_threshold')
parser.add_argument('--nsn-trf-sc', type=str, nargs='+',
                    help='NanoStringNormalizer sample_content')
parser.add_argument('--rfe-srv-step', type=float, nargs='+',
                    help='RFE step')
parser.add_argument('--rfe-srv-tune-step-at', type=int,
                    help='RFE tune step at')
parser.add_argument('--rfe-srv-reducing-step', default=False,
                    action='store_true', help='RFE reducing step')
parser.add_argument('--rfe-srv-verbose', type=int, default=0,
                    help='RFE verbosity')
parser.add_argument('--cph-srv-am', type=int, default=1,
                    help='CoxPHSurvivalAnalysis alpha mantissa')
parser.add_argument('--cph-srv-ae', type=int, nargs='+',
                    help='CoxPHSurvivalAnalysis alpha exp')
parser.add_argument('--cph-srv-ae-min', type=int,
                    help='CoxPHSurvivalAnalysis alpha exp min')
parser.add_argument('--cph-srv-ae-max', type=int,
                    help='CoxPHSurvivalAnalysis alpha exp max')
parser.add_argument('--cph-srv-num-alphas', type=int, default=10,
                    help='CoxPHSurvivalAnalysis num alphas')
parser.add_argument('--cph-srv-ties', type=str, default='efron',
                    help='CoxPHSurvivalAnalysis ties')
parser.add_argument('--cph-srv-n-iter', type=int, default=1000,
                    help='CoxPHSurvivalAnalysis n_iter')
parser.add_argument('--cph-srv-base-alpha', type=float, default=1e-5,
                    help='CoxPHSurvivalAnalysis base_alpha')
parser.add_argument('--cph-srv-max-iter', type=lambda x: int(float(x)),
                    default=1e5,
                    help='FastCoxPHSurvivalAnalysis max_iter')
parser.add_argument('--cph-srv-fit-baseline-model', default=False,
                    action='store_true',
                    help='FastCoxPHSurvivalAnalysis fit_baseline_model')
parser.add_argument('--cph-srv-normalize', default=False, action='store_true',
                    help='FastCoxPHSurvivalAnalysis normalize')
parser.add_argument('--cnet-srv-l1r', type=float, nargs='+',
                    help='CoxnetSurvivalAnalysis l1_ratio')
parser.add_argument('--cnet-srv-l1r-min', type=float,
                    help='CoxnetSurvivalAnalysis l1_ratio min')
parser.add_argument('--cnet-srv-l1r-max', type=float,
                    help='CoxnetSurvivalAnalysis l1_ratio max')
parser.add_argument('--cnet-srv-l1r-step', type=float, default=0.05,
                    help='CoxnetSurvivalAnalysis l1_ratio step')
parser.add_argument('--cnet-srv-na', type=int, nargs='+',
                    help='CoxnetSurvivalAnalysis n_alphas')
parser.add_argument('--cnet-srv-alpha-min-ratio', type=float, default=0.1,
                    help='CoxnetSurvivalAnalysis alpha_min_ratio')
parser.add_argument('--cnet-srv-max-iter', type=lambda x: int(float(x)),
                    default=1e5,
                    help='CoxnetSurvivalAnalysis max_iter')
parser.add_argument('--cnet-srv-fit-baseline-model', default=False,
                    action='store_true',
                    help='CoxnetSurvivalAnalysis fit_baseline_model')
parser.add_argument('--cnet-srv-normalize', default=False, action='store_true',
                    help='CoxnetSurvivalAnalysis normalize')
parser.add_argument('--fsvm-srv-ae', type=int, nargs='+',
                    help='FastSurvivalSVM alpha exp')
parser.add_argument('--fsvm-srv-ae-min', type=int,
                    help='FastSurvivalSVM alpha exp min')
parser.add_argument('--fsvm-srv-ae-max', type=int,
                    help='FastSurvivalSVM alpha exp max')
parser.add_argument('--fsvm-srv-rr', type=float, nargs='+',
                    help='FastSurvivalSVM rank_ratio')
parser.add_argument('--fsvm-srv-rr-min', type=float,
                    help='FastSurvivalSVM rank_ratio min')
parser.add_argument('--fsvm-srv-rr-max', type=float,
                    help='FastSurvivalSVM rank_ratio max')
parser.add_argument('--fsvm-srv-rr-step', type=float, default=0.05,
                    help='FastSurvivalSVM rank_ratio step')
parser.add_argument('--fsvm-srv-o', type=str, nargs='+',
                    choices=['avltree', 'direct-count', 'PRSVM', 'rbtree',
                             'simple'],
                    help='FastSurvivalSVM optimizer')
parser.add_argument('--fsvm-srv-max-iter', type=int, default=20,
                    help='FastSurvivalSVM max_iter')
parser.add_argument('--edger-no-log', default=False,
                    action='store_true',
                    help='edger no log transform')
parser.add_argument('--nano-meta-col', type=str, default='Code.Class',
                    help='NanoString Code Class feature metadata column name')
parser.add_argument('--scv-type', type=str,
                    choices=['grid', 'rand'], default='grid',
                    help='scv type')
parser.add_argument('--scv-splits', type=int, default=10,
                    help='scv splits')
parser.add_argument('--scv-repeats', type=int, default=0,
                    help='scv repeats')
parser.add_argument('--scv-size', type=float, default=0.2,
                    help='scv size')
parser.add_argument('--scv-verbose', type=int,
                    help='scv verbosity')
parser.add_argument('--scv-scoring', type=str, nargs='+',
                    choices=['concordance_index_censored',
                             'concordance_index_ipcw',
                             'cumulative_dynamic_auc'],
                    help='scv scoring metric')
parser.add_argument('--scv-refit', type=str,
                    choices=['concordance_index_censored',
                             'concordance_index_ipcw',
                             'cumulative_dynamic_auc'],
                    help='scv refit scoring metric')
parser.add_argument('--scv-n-iter', type=int, default=100,
                    help='randomized scv num iterations')
parser.add_argument('--scv-error-score', type=str, default='nan',
                    help='scv error score')
parser.add_argument('--scv-use-ssplit', default=False, action='store_true',
                    help='scv ShuffleSplit variants instead of KFold')
parser.add_argument('--test-splits', type=int, default=10,
                    help='num outer splits')
parser.add_argument('--test-repeats', type=int, default=0,
                    help='num outer repeats')
parser.add_argument('--test-size', type=float, default=0.2,
                    help='outer splits test size')
parser.add_argument('--test-use-ssplit', default=False, action='store_true',
                    help='test ShuffleSplit variants instead of KFold')
parser.add_argument('--param-cv-score-meth', type=str,
                    choices=['best', 'all'], default='best',
                    help='param cv scores calculation method')
parser.add_argument('--feature-rank-meth', type=str,
                    choices=['num_select_plus1', 'num_total'],
                    default='num_select_plus1',
                    help='feature rank method')
parser.add_argument('--sns-color-palette', type=str, default='hls',
                    help='Seaborn/matplotlib color palette')
parser.add_argument('--title-font-size', type=int, default=14,
                    help='figure title font size')
parser.add_argument('--axis-font-size', type=int, default=14,
                    help='figure axis font size')
parser.add_argument('--long-label-names', default=False, action='store_true',
                    help='figure long label names')
parser.add_argument('--fig-width', type=float, default=10,
                    help='figure width')
parser.add_argument('--fig-height', type=float, default=10,
                    help='figure height')
parser.add_argument('--fig-format', type=str, nargs='+',
                    choices=['png', 'pdf', 'svg', 'tif'], default=['png'],
                    help='figure format')
parser.add_argument('--save-figs', default=False, action='store_true',
                    help='save figures')
parser.add_argument('--show-figs', default=False, action='store_true',
                    help='show figures')
parser.add_argument('--save-models', default=False, action='store_true',
                    help='save models')
parser.add_argument('--save-results', default=False, action='store_true',
                    help='save results')
parser.add_argument('--save-model-code', type=str,
                    help='save model code')
parser.add_argument('--n-jobs', type=int, default=-1,
                    help='num parallel jobs')
parser.add_argument('--parallel-backend', type=str, default='loky',
                    help='joblib parallel backend')
parser.add_argument('--max-nbytes', type=str, default='1M',
                    help='joblib parallel max_nbytes')
parser.add_argument('--pipe-memory', default=False, action='store_true',
                    help='turn on pipeline memory')
parser.add_argument('--gbytes-limit', type=int,
                    help='joblib memory cache size limit in GB')
parser.add_argument('--out-dir', type=dir_path, default=os.getcwd(),
                    help='output dir')
parser.add_argument('--tmp-dir', type=dir_path, default=gettempdir(),
                    help='tmp dir')
parser.add_argument('--random-seed', type=int, default=777,
                    help='random state seed')
parser.add_argument('--jvm-heap-size', type=int, default=500,
                    help='rjava jvm heap size')
parser.add_argument('--filter-warnings', type=str, nargs='+',
                    choices=['convergence', 'joblib', 'fitfailed', 'coxnet'],
                    help='filter warnings')
parser.add_argument('--verbose', type=int, default=1,
                    help='program verbosity')
parser.add_argument('--load-only', default=False, action='store_true',
                    help='set up model selection and load dataset only')

num_col_trfs = 0
args, argv = parser.parse_known_args()
if args.pipe_steps[0] == ['ColumnTransformer']:
    args, argv, num_col_trfs = add_col_trf_args(args, argv)
if argv:
    parser.error('unrecognized arguments: {}'.format(' '.join(argv)))

if args.test_size >= 1.0:
    args.test_size = int(args.test_size)
if args.scv_size >= 1.0:
    args.scv_size = int(args.scv_size)
if args.scv_error_score.isdigit():
    args.scv_error_score = int(args.scv_error_score)
elif args.scv_error_score == 'nan':
    args.scv_error_score = np.nan
if args.scv_verbose is None:
    args.scv_verbose = args.verbose
if args.scv_scoring is None:
    args.scv_refit = 'score'
    args.scv_scoring = [args.scv_refit]
if args.max_nbytes == 'None':
    args.max_nbytes = None
if args.gbytes_limit == 'None':
    args.gbytes_limit = None

if args.parallel_backend != 'multiprocessing':
    python_warnings = ([os.environ['PYTHONWARNINGS']]
                       if 'PYTHONWARNINGS' in os.environ else [])
    python_warnings.append(':'.join(
        ['ignore', '', 'FutureWarning', 'sklearn.utils.deprecation']))
    python_warnings.append(':'.join(
        ['ignore', '', 'FutureWarning', 'rpy2.robjects.pandas2ri']))
    os.environ['PYTHONWARNINGS'] = ','.join(python_warnings)
if args.filter_warnings:
    if args.parallel_backend == 'multiprocessing':
        if 'convergence' in args.filter_warnings:
            # filter convergence warnings
            warnings.filterwarnings(
                'ignore', category=ConvergenceWarning,
                message='^Optimization (did not converge|terminated early)')
        if 'joblib' in args.filter_warnings:
            # filter joblib peristence time warnings
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message='^Persisting input arguments took')
        if any(w in args.filter_warnings for w in ('fitfailed', 'coxnet')):
            warnings.filterwarnings(
                'ignore', category=FitFailedWarning,
                message='^Estimator fit failed')
        if 'coxnet' in args.filter_warnings:
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message='^all coefficients are zero',
                module='sksurv_extensions.linear_model._coxnet')
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning,
                message='^divide by zero encountered in true_divide',
                module='sksurv.linear_model.coxph')
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning,
                message='^invalid value encountered in true_divide')
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning,
                message='^overflow encountered in exp',
                module='sksurv.linear_model.coxph')
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning,
                message='^overflow encountered in power',
                module='sksurv.linear_model.coxph')
    else:
        python_warnings = ([os.environ['PYTHONWARNINGS']]
                           if 'PYTHONWARNINGS' in os.environ else [])
        if 'convergence' in args.filter_warnings:
            python_warnings.append(':'.join(
                ['ignore', 'Optimization did not converge', 'UserWarning']))
            python_warnings.append(':'.join(
                ['ignore', 'Optimization terminated early', 'UserWarning']))
        if 'joblib' in args.filter_warnings:
            python_warnings.append(':'.join(
                ['ignore', 'Persisting input arguments took', 'UserWarning']))
        if any(w in args.filter_warnings for w in ('fitfailed', 'coxnet')):
            python_warnings.append(':'.join(
                ['ignore', 'Estimator fit failed', 'RuntimeWarning']))
        if 'coxnet' in args.filter_warnings:
            python_warnings.append(':'.join(
                ['ignore', 'all coefficients are zero', 'UserWarning',
                 'sksurv_extensions.linear_model._coxnet']))
            python_warnings.append(':'.join(
                ['ignore', 'divide by zero encountered in true_divide',
                 'RuntimeWarning', 'sksurv.linear_model.coxph']))
            python_warnings.append(':'.join(
                ['ignore', 'invalid value encountered in true_divide',
                 'RuntimeWarning']))
            python_warnings.append(':'.join(
                ['ignore', 'overflow encountered in exp', 'RuntimeWarning',
                 'sksurv.linear_model.coxph']))
            python_warnings.append(':'.join(
                ['ignore', 'overflow encountered in power', 'RuntimeWarning',
                 'sksurv.linear_model.coxph']))
        os.environ['PYTHONWARNINGS'] = ','.join(python_warnings)

inner_max_num_threads = 1 if args.parallel_backend in ('loky') else None
if args.max_nbytes is None:
    joblib_temp_folder_mgr = TemporaryResourcesManager()

# suppress linux conda qt5 wayland warning
if sys.platform.startswith('linux'):
    os.environ['XDG_SESSION_TYPE'] = 'x11'

r_base = importr('base')
r_biobase = importr('Biobase')
robjects.r('set.seed({:d})'.format(args.random_seed))
robjects.r('options(\'java.parameters\'="-Xmx{:d}m")'
           .format(args.jvm_heap_size))

atexit.register(run_cleanup)

if args.pipe_memory:
    cachedir = mkdtemp(dir=args.tmp_dir)
    bytes_limit = (args.gbytes_limit * 1024 ** 3
                   if args.gbytes_limit is not None else None)
    memory = Memory(location=cachedir, verbose=0, bytes_limit=bytes_limit)
    cnet_srv = CachedExtendedCoxnetSurvivalAnalysis(
        memory=memory, alpha_min_ratio=args.cnet_srv_alpha_min_ratio,
        fit_baseline_model=args.cnet_srv_fit_baseline_model,
        max_iter=args.cnet_srv_max_iter, normalize=args.cnet_srv_normalize)
else:
    memory = None
    cnet_srv = ExtendedCoxnetSurvivalAnalysis(
        alpha_min_ratio=args.cnet_srv_alpha_min_ratio,
        fit_baseline_model=args.cnet_srv_fit_baseline_model,
        max_iter=args.cnet_srv_max_iter, normalize=args.cnet_srv_normalize)

pipeline_step_types = ('slr', 'trf', 'srv')
cv_params = {k: v for k, v in vars(args).items()
             if '_' in k and k.split('_')[1] in pipeline_step_types}
if cv_params['col_slr_file']:
    for feature_file in cv_params['col_slr_file']:
        if not os.path.isfile(feature_file):
            raise IOError('File does not exist/invalid: {}'
                          .format(feature_file))
        with open(feature_file) as f:
            feature_names = f.read().splitlines()
        feature_names = [n.strip() for n in feature_names]
        if cv_params['col_slr_cols'] is None:
            cv_params['col_slr_cols'] = []
        cv_params['col_slr_cols'].append(feature_names)
for cv_param, cv_param_values in cv_params.copy().items():
    if cv_param_values is None:
        if cv_param in ('cph_srv_ae', 'fsvm_srv_ae'):
            cv_params[cv_param[:-1]] = None
        continue
    if cv_param in ('col_slr_cols', 'cft_slr_thres', 'mnt_slr_thres',
                    'mdt_slr_thres', 'skb_slr_k', 'log_trf_shift',
                    'rna_trf_pc', 'rfe_srv_step', 'cnet_srv_l1r',
                    'cnet_srv_na', 'fsvm_srv_rr'):
        cv_params[cv_param] = np.sort(cv_param_values, kind='mergesort')
    elif cv_param in ('rna_slr_mb', 'rna_trf_ft', 'rna_trf_mb', 'nsn_trf_cc',
                      'nsn_trf_bg', 'nsn_trf_bg_t', 'nsn_trf_sc',
                      'pwr_trf_meth', 'fsvm_srv_o'):
        cv_params[cv_param] = sorted(cv_param_values)
    elif cv_param == 'skb_slr_k_max':
        cv_param = '_'.join(cv_param.split('_')[:3])
        if (cv_params['{}_min'.format(cv_param)] == 1
                and cv_params['{}_step'.format(cv_param)] > 1):
            cv_params[cv_param] = np.array([1] + list(range(
                0, cv_params['{}_max'.format(cv_param)]
                + cv_params['{}_step'.format(cv_param)],
                cv_params['{}_step'.format(cv_param)]))[1:])
        else:
            cv_params[cv_param] = np.array(range(
                cv_params['{}_min'.format(cv_param)],
                cv_params['{}_max'.format(cv_param)]
                + cv_params['{}_step'.format(cv_param)],
                cv_params['{}_step'.format(cv_param)]))
    elif cv_param == 'cph_srv_ae':
        cv_params[cv_param[:-1]] = (
            cv_params['cph_srv_am'] * 10. ** np.asarray(cv_param_values))
    elif cv_param == 'cph_srv_ae_max':
        cv_param = '_'.join(cv_param.split('_')[:-1])
        cv_param_v_min = (cv_params['cph_srv_am'] * 10.
                          ** cv_params['{}_min'.format(cv_param)])
        cv_param_v_max = cv_params['cph_srv_am'] * 10. ** cv_param_values
        cv_params[cv_param[:-1]] = np.logspace(np.log10(cv_param_v_min),
                                               np.log10(cv_param_v_max),
                                               cv_params['cph_srv_num_alphas'])
    elif cv_param == 'fsvm_srv_ae':
        cv_params[cv_param[:-1]] = 2. ** np.asarray(cv_param_values)
    elif cv_param == 'fsvm_srv_ae_max':
        cv_param = '_'.join(cv_param.split('_')[:-1])
        cv_param_v_min = cv_params['{}_min'.format(cv_param)]
        cv_param_v_max = cv_param_values
        cv_params[cv_param[:-1]] = np.logspace(
            cv_param_v_min, cv_param_v_max,
            cv_param_v_max - cv_param_v_min + 1, base=2)
    elif cv_param in ('cnet_srv_l1r_max', 'fsvm_srv_rr_max'):
        cv_param = '_'.join(cv_param.split('_')[:3])
        cv_params[cv_param] = np.round(
            np.linspace(cv_params['{}_min'.format(cv_param)],
                        cv_params['{}_max'.format(cv_param)],
                        round((cv_params['{}_max'.format(cv_param)]
                               - cv_params['{}_min'.format(cv_param)])
                              / cv_params['{}_step'.format(cv_param)])
                        + 1), decimals=3)
cnet_srv_min_1lr = 1e-7
if cv_params['cnet_srv_l1r'] is not None and any(
        r < cnet_srv_min_1lr for r in cv_params['cnet_srv_l1r']):
    cv_params['cnet_srv_l1r'] = np.array(
        [cnet_srv_min_1lr] + [r for r in cv_params['cnet_srv_l1r']
                              if r >= cnet_srv_min_1lr])

pipe_config = {
    # feature selectors
    'ColumnSelector': {
        'estimator': ColumnSelector(meta_col=args.col_slr_meta_col),
        'param_grid': {
            'cols': cv_params['col_slr_cols']},
        'param_routing': ['feature_meta']},
    'ConfidenceThreshold': {
        'estimator': ConfidenceThreshold(meta_col=args.cft_slr_meta_col),
        'param_grid': {
            'threshold': cv_params['cft_slr_thres']},
        'param_routing': ['feature_meta']},
    'MeanThreshold': {
        'estimator': MeanThreshold(),
        'param_grid': {
            'threshold': cv_params['mnt_slr_thres']}},
    'MedianThreshold': {
        'estimator': MedianThreshold(),
        'param_grid': {
            'threshold': cv_params['mdt_slr_thres']}},
    'EdgeRFilterByExpr': {
        'estimator': EdgeRFilterByExpr(is_classif=False),
        'param_grid': {
            'model_batch': cv_params['rna_slr_mb']},
        'param_routing': ['sample_meta']},
    'NanoStringEndogenousSelector': {
        'estimator': NanoStringEndogenousSelector(meta_col=args.nano_meta_col),
        'param_routing': ['feature_meta']},
    # transformers
    'ColumnTransformer': {
        'estimator': ExtendedColumnTransformer([], n_jobs=1)},
    'OneHotEncoder': {
        'estimator': OneHotEncoder(
            categories=(args.ohe_trf_categories if args.ohe_trf_categories else
                        'auto'), drop=args.ohe_trf_drop,
            handle_unknown=('ignore' if args.ohe_trf_drop is None else
                            'error'),
            sparse=False)},
    'OrdinalEncoder': {
        'estimator': OrdinalEncoder(
            categories=(args.ord_trf_categories if args.ord_trf_categories else
                        'auto'))},
    'LogTransformer': {
        'estimator': LogTransformer(base=args.log_trf_base),
        'param_grid': {
            'shift': cv_params['log_trf_shift']}},
    'PowerTransformer': {
        'estimator': PowerTransformer(),
        'param_grid': {
            'method': cv_params['pwr_trf_meth']}},
    'MinMaxScaler': {
        'estimator': MinMaxScaler(feature_range=args.mms_trf_feature_range)},
    'RobustScaler': {
        'estimator': RobustScaler()},
    'StandardScaler': {
        'estimator': StandardScaler()},
    'DESeq2RLEVST': {
        'estimator': DESeq2RLEVST(is_classif=False),
        'param_grid': {
            'fit_type': cv_params['rna_trf_ft'],
            'model_batch': cv_params['rna_trf_mb']},
        'param_routing': ['sample_meta']},
    'EdgeRTMMCPM': {
        'estimator': EdgeRTMMCPM(log=not args.edger_no_log),
        'param_grid': {
            'prior_count': cv_params['rna_trf_pc']},
        'param_routing': ['sample_meta']},
    'EdgeRTMMTPM': {
        'estimator': EdgeRTMMTPM(log=not args.edger_no_log),
        'param_grid': {
            'prior_count': cv_params['rna_trf_pc']},
        'param_routing': ['feature_meta']},
    'LimmaBatchEffectRemover': {
        'estimator': LimmaBatchEffectRemover(preserve_design=False),
        'param_routing': ['sample_meta']},
    'NanoStringNormalizer': {
        'estimator': NanoStringNormalizer(meta_col=args.nano_meta_col),
        'param_grid': {
            'code_count': cv_params['nsn_trf_cc'],
            'background': cv_params['nsn_trf_bg'],
            'background_threshold': cv_params['nsn_trf_bg_t'],
            'sample_content': cv_params['nsn_trf_sc']},
        'param_routing': ['feature_meta']},
    'NanoStringDiffNormalizer': {
        'estimator': NanoStringDiffNormalizer(meta_col=args.nano_meta_col),
        'param_routing': ['feature_meta']},
    # survival analyzers
    'SelectFromUnivariateSurvivalModel-CoxPHSurvivalAnalysis': {
        'estimator': SelectFromUnivariateSurvivalModel(
            ExtendedCoxPHSurvivalAnalysis(ties=args.cph_srv_ties,
                                          n_iter=args.cph_srv_n_iter,
                                          base_alpha=args.cph_srv_base_alpha),
            memory=memory),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'estimator__alpha': cv_params['cph_srv_a']},
        'param_routing': ['feature_meta']},
    'SelectFromUnivariateSurvivalModel-FastCoxPHSurvivalAnalysis': {
        'estimator': SelectFromUnivariateSurvivalModel(
            FastCoxPHSurvivalAnalysis(
                fit_baseline_model=args.cph_srv_fit_baseline_model,
                max_iter=args.cph_srv_max_iter,
                normalize=args.cph_srv_normalize),
            memory=memory),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'estimator__alpha': cv_params['cph_srv_a']},
        'param_routing': ['feature_meta']},
    'SelectFromUnivariateSurvivalModel-FastSurvivalSVM': {
        'estimator': SelectFromUnivariateSurvivalModel(
            FastSurvivalSVM(max_iter=args.fsvm_srv_max_iter,
                            random_state=args.random_seed), memory=memory),
        'param_grid': {
            'k': cv_params['skb_slr_k'],
            'estimator__alpha': cv_params['fsvm_srv_a'],
            'estimator__rank_ratio': cv_params['fsvm_srv_rr'],
            'estimator__optimizer': cv_params['fsvm_srv_o']},
        'param_routing': ['feature_meta']},
    'RFE-FastSurvivalSVM': {
        'estimator': ExtendedRFE(
            FastSurvivalSVM(max_iter=args.fsvm_srv_max_iter,
                            random_state=args.random_seed),
            tune_step_at=args.rfe_srv_tune_step_at,
            reducing_step=args.rfe_srv_reducing_step,
            verbose=args.rfe_srv_verbose, memory=memory),
        'param_grid': {
            'estimator__alpha': cv_params['fsvm_srv_a'],
            'estimator__rank_ratio': cv_params['fsvm_srv_rr'],
            'estimator__optimizer': cv_params['fsvm_srv_o'],
            'step': cv_params['rfe_srv_step'],
            'n_features_to_select': cv_params['skb_slr_k']},
        'param_routing': ['feature_meta']},
    'CoxPHSurvivalAnalysis': {
        'estimator': ExtendedCoxPHSurvivalAnalysis(
            ties=args.cph_srv_ties, n_iter=args.cph_srv_n_iter,
            base_alpha=args.cph_srv_base_alpha),
        'param_grid': {
            'alpha': cv_params['cph_srv_a']},
        'param_routing': ['feature_meta']},
    'FastCoxPHSurvivalAnalysis': {
        'estimator': FastCoxPHSurvivalAnalysis(
            fit_baseline_model=args.cph_srv_fit_baseline_model,
            max_iter=args.cph_srv_max_iter, normalize=args.cph_srv_normalize),
        'param_grid': {
            'alpha': cv_params['cph_srv_a']},
        'param_routing': ['feature_meta']},
    'CoxnetSurvivalAnalysis': {
        'estimator': MetaCoxnetSurvivalAnalysis(cnet_srv),
        'param_grid': {
            'estimator__l1_ratio': cv_params['cnet_srv_l1r'],
            'estimator__n_alphas': cv_params['cnet_srv_na']},
        'param_routing': ['feature_meta']},
    'FastSurvivalSVM': {
        'estimator': FastSurvivalSVM(max_iter=args.fsvm_srv_max_iter,
                                     random_state=args.random_seed),
        'param_grid': {
            'alpha': cv_params['fsvm_srv_a'],
            'rank_ratio': cv_params['fsvm_srv_rr'],
            'optimizer': cv_params['fsvm_srv_o']}}}

params_lin_xticks = [
    'trf__prior_count',
    'trf__shift',
    'srv__k',
    'srv__n_features_to_select',
    'srv__step',
    'srv__l1_ratio',
    'srv__n_alphas',
    'srv__rank_ratio',
    'srv__estimator__l1_ratio',
    'srv__estimator__n_alphas',
    'srv__estimator__rank_ratio']

params_log_xticks = [
    'srv__alpha',
    'srv__alphas',
    'srv__estimator__alpha']

params_fixed_xticks = [
    'slr',
    'slr__cols',
    'slr__model_batch',
    'slr__threshold',
    'trf',
    'trf__method',
    'trf__model_batch',
    'trf__code_count',
    'trf__background',
    'trf__sample_content',
    'srv',
    'srv__estimator__optimizer',
    'srv__optimizer']

params_k_selected_features = [
    'slr__k',
    'slr__max_features',
    'clf__n_features_to_select']

metric_label = {
    'score': 'C-index',
    'concordance_index_censored': 'C-index',
    'concordance_index_ipcw': 'IPCW C-index',
    'cumulative_dynamic_auc': 'CD ROC AUC'}

ordinal_encoder_categories = {
    'tumor_stage': ['0', 'i', 'i or ii', 'ii', 'NA', 'iii', 'iv']}

run_model_selection()
if args.show_figs or args.save_figs:
    for fig_num in plt.get_fignums():
        plt.figure(fig_num, constrained_layout=True)
        if args.save_figs:
            for fig_fmt in args.fig_format:
                plt.savefig('{}/Figure_{:d}.{}'.format(args.out_dir, fig_num,
                                                       fig_fmt),
                            bbox_inches='tight', format=fig_fmt)
if args.show_figs:
    plt.show()
