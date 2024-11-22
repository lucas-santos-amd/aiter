# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: test_common.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-03 15:53:32
# @Last Modified By: valarLip
# @Last Modified At: 2024-11-22 16:49:03
# @Description: This is description.

import torch
import numpy as np
import pandas as pd
from ater import logger
import torch.profiler as tpf
num_iters = 100
num_warmup = 20


def perftest(name=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tpf.profile(activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                             profile_memory=True,
                             with_stack=True,
                             with_modules=True,
                             record_shapes=True,
                             schedule=tpf.schedule(wait=1,
                                                   warmup=num_warmup,
                                                   active=num_iters),) as prof:
                for _ in range(1+num_iters+num_warmup):
                    data = func(*args, **kwargs)
                    prof.step()
            avg = get_trace_perf(prof)
            return data, avg
        return wrapper
    return decorator


def get_trace_perf(prof):
    print(vars(prof.key_averages()))
    df = []
    for el in prof.key_averages():
        if 'ProfilerStep*' not in el.key:
            df.append(vars(el))
    df = pd.DataFrame(df)
    cols = ['key', 'count',
            'cpu_time_total', 'self_cpu_time_total',
            'device_time_total', 'self_device_time_total',
            'self_device_memory_usage',
            'device_type',]
    cols = [el for el in df.columns if el in cols]
    df = df[(df.self_cpu_time_total > 0) | (df.self_device_time_total > 0)]

    timerList = ['self_cpu_time_total', 'self_device_time_total', ]
    df = df[cols].sort_values(timerList)
    avg_name = '[avg ms/iter]'
    for el in timerList:
        df.at[avg_name, el] = df[el].sum()/num_iters
    logger.info(f'{df}')
    return df.at[avg_name, 'self_device_time_total']


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg=''):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        logger.info(f'{msg}[passed~]')
    else:
        percent = (a[mask]).numel()/a.numel()
        if percent > 0.01:
            logger.info(f'''{msg}[failed!]
        a:  {a.shape}
            {a[mask]}
        b:  {b.shape}
            {b[mask]}
    dtlta:
            {(a-b)[mask]}''')
        else:
            logger.info(
                f'''{msg}[waring!] a and b results are not all close''')
        logger.info(
            f'-->max delta:{(a-b).max()}, delta details: {percent:.1%} ({(a[mask]).numel()} of {a.numel()}) elements {atol=} {rtol=}')
