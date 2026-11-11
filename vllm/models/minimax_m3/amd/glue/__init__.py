# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1-Flash TP4 MoE-half paths (overlay):

* decode glue (VLLM_M3_DECODE_GLUE=1): pure-decode steps of 17 .. 192 rows, one
  FlyDSL launch from the o_proj partial to the MoE inputs (kernel.py, decode.py);
* SP prefill (VLLM_M3_SP_PREFILL=1): eager steps of VLLM_M3_SP_MIN_ROWS+ rows,
  sequence-parallel norm / router / shared expert / quant (sp.py).

Both are off unless enabled; see runtime.py for the lifecycle.
"""

from .runtime import M3Glue, StepPlan

__all__ = ["M3Glue", "StepPlan"]
