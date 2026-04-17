# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# M110: DES-LOC coordination for Domino async tensor parallel
DESLOC_DOMINO_OVERLAP_ENABLED = False

def set_desloc_domino_overlap(enabled):
    """Toggle overlap of DES-LOC AllReduce with Domino TP comm."""
    global DESLOC_DOMINO_OVERLAP_ENABLED
    DESLOC_DOMINO_OVERLAP_ENABLED = enabled

def get_desloc_domino_overlap():
    return DESLOC_DOMINO_OVERLAP_ENABLED
