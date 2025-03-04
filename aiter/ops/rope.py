# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor, empty, empty_like, autograd
from typing import Tuple, Union
from ..jit.core import compile_ops


MD_NAME = "module_rope"


@compile_ops("module_rope")
def rope_fwd_impl(
    output: Tensor,
    input: Tensor,
    freqs: Tensor
): 
    '''
    Forward propagation of traditional RoPE (Rotary Position Embedding).
    Input and output should be in "sbhd" format and freqs should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_fwd(
    input: Tensor,
    freqs: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = input.shape
    output = empty((b, s, h, d), dtype=input.dtype, device=input.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(input, requires_grad=False)
    rope_fwd_impl(output, input, freqs)
    return output

@compile_ops("module_rope")
def rope_bwd_impl(
    input_grads: Tensor,
    output_grads: Tensor,
    freqs: Tensor
): 
    '''
    Backward propagation of traditional RoPE (Rotary Position Embedding).
    Input and output should be in "sbhd" format and freqs should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_bwd(
    output_grads: Tensor,
    freqs: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = output_grads.shape
    input_grads = empty((b, s, h, d), dtype=output_grads.dtype, device=output_grads.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(output_grads, requires_grad=False)
    rope_bwd_impl(input_grads, output_grads, freqs)
    return input_grads


class RoPE(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, freqs: Tensor, transpose_output: bool = False) -> Tensor:
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(freqs)
        return rope_fwd(x, freqs, transpose_output)
    
    @staticmethod
    def backward(ctx, output_grads: Tensor) -> Tuple[Union[Tensor, None], ...]:
        (freqs,) = ctx.saved_tensors
        return rope_bwd(output_grads, freqs, ctx.transpose_output), None, None


@compile_ops("module_rope")
def rope_cached_fwd_impl(
    output: Tensor,
    input: Tensor,
    cos: Tensor,
    sin: Tensor
): 
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with cached cos and sin.
    Input and output should be in "sbhd" format, and cos and sin should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_cached_fwd(
    input: Tensor,
    cos: Tensor,
    sin: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = input.shape
    output = empty((b, s, h, d), dtype=input.dtype, device=input.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(input, requires_grad=False)
    rope_cached_fwd_impl(output, input, cos, sin)
    return output

@compile_ops("module_rope")
def rope_cached_bwd_impl(
    input_grads: Tensor,
    output_grads: Tensor,
    cos: Tensor,
    sin: Tensor
): 
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with cached cos and sin.
    Input and output should be in "sbhd" format, and cos and sin should be in shape of [s, 1, 1, d].
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_cached_bwd(
    output_grads: Tensor,
    cos: Tensor,
    sin: Tensor,
    transpose_output: bool = False
) -> Tensor :
    s, b, h, d = output_grads.shape
    input_grads = empty((b, s, h, d), dtype=output_grads.dtype, device=output_grads.device, requires_grad=False).transpose(0, 1)\
        if transpose_output else empty_like(output_grads, requires_grad=False)
    rope_cached_bwd_impl(input_grads, output_grads, cos, sin)
    return input_grads


class RoPECached(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor, transpose_output: bool = False) -> Tensor:
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(cos, sin)
        return rope_cached_fwd(x, cos, sin, transpose_output)
    
    @staticmethod
    def backward(ctx, output_grads) -> Tuple[Union[Tensor, None], ...]:
        cos, sin = ctx.saved_tensors
        return rope_cached_bwd(output_grads, cos, sin, ctx.transpose_output), None, None


@compile_ops("module_rope")
def rope_thd_fwd_impl(
    output: Tensor,
    input: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
):
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with input sizes: (t, h, d).
    where t is cumulative sum of sequence lengths.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_thd_fwd(
    input: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
) -> Tensor :
    output = empty_like(input, requires_grad=False)
    rope_thd_fwd_impl(output, input, cu_seqlens, freqs)
    return output

@compile_ops("module_rope")
def rope_thd_bwd_impl(
    input_grads: Tensor,
    output_grads: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
):
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with input sizes: (t, h, d).
    where t is cumulative sum of sequence lengths.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_thd_bwd(
    output_grads: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
) -> Tensor :
    input_grads = empty_like(output_grads, requires_grad=False)
    rope_thd_bwd_impl(input_grads, output_grads, cu_seqlens, freqs)
    return input_grads


class RoPETHD(autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, cu_seqlens: Tensor, freqs: Tensor):
        ctx.save_for_backward(cu_seqlens, freqs)
        return rope_thd_fwd(x, cu_seqlens, freqs)
    
    @staticmethod
    def backward(ctx, output_grads) -> Tuple[Union[Tensor, None], ...]:
        cu_seqlens, freqs = ctx.saved_tensors
        return rope_thd_bwd(output_grads, cu_seqlens, freqs), None, None


@compile_ops("module_rope")
def rope_2d_fwd_impl(
    output: Tensor,
    input: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    img_height: int,
    img_width: int,
):
    '''
    Forward propagation of RoPE (Rotary Position Embedding) with 2D image as input.
    Input and output should be in (b, s, h, d) where s = H * W.
    cos_h and sin_h are in (1, H', 1, h, d // 2) where H' >= H.
    cos_w and sin_w are in (1, 1, W', h, d // 2) where W' >= W.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_2d_fwd(
    input: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    img_height: int,
    img_width: int
) -> Tensor :
    output = empty_like(input, requires_grad=False)
    rope_2d_fwd_impl(output, input, cos_h, sin_h, cos_w, sin_w, img_height, img_width)
    return output

@compile_ops("module_rope")
def rope_2d_bwd_impl(
    input_grads: Tensor,
    output_grads: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    img_height: int,
    img_width: int,
):
    '''
    Backward propagation of RoPE (Rotary Position Embedding) with 2D image as input.
    output_grads and input_grads should be in (b, s, h, d) where s = H * W.
    cos_h and sin_h are in (1, H', 1, h, d // 2) where H' >= H.
    cos_w and sin_w are in (1, 1, W', h, d // 2) where W' >= W.
    This implemenation rotates the 2nd half of elements.
    '''
    ...

def rope_2d_bwd(
    output_grads: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    img_height: int,
    img_width: int
) -> Tensor :
    input_grads = empty_like(output_grads, requires_grad=False)
    rope_2d_bwd_impl(input_grads, output_grads, cos_h, sin_h, cos_w, sin_w, img_height, img_width)
    return input_grads


class RoPE2D(autograd.Function):
    @staticmethod
    def forward(ctx, x:
                Tensor, cos_height: Tensor, sin_height:
                Tensor, cos_width: Tensor, sin_width: Tensor,
                img_height: int, img_width: int) -> Tensor:
        ctx.save_for_backward(cos_height, sin_height, cos_width, sin_width)
        ctx.img_height = img_height
        ctx.img_width = img_width
        return rope_2d_fwd(x, cos_height, sin_height, cos_width, sin_width, img_height, img_width)
    
    @staticmethod
    def backward(ctx, output_grads) -> Tuple[Union[Tensor, None], ...]:
        cos_height, sin_height, cos_width, sin_width = ctx.saved_tensors
        return rope_2d_bwd(output_grads, cos_height, sin_height, cos_width, sin_width, ctx.img_height, ctx.img_height), None, None
