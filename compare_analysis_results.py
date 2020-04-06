#!/usr/bin/env python3
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
from os import fspath
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import scipy.sparse

from set_ops import sorted_union

@dataclass
class CellByBinMatrix:
    matrix: scipy.sparse.spmatrix
    barcodes: List[str]
    bins: List[str]

def read_pipeline_output(base_directory: Path):
    with open(base_directory / 'barcodes.txt') as f:
        barcodes = [line.strip().split('_')[-1] for line in f]
    with open(base_directory / 'bins.txt') as f:
        bins = [line.strip() for line in f]

    # TODO: drop 'fspath' call after  https://github.com/scipy/scipy/pull/11439 is merged
    cell_by_bin = scipy.io.mmread(fspath(base_directory / 'filtered_cell_by_bin.mtx'))

    return CellByBinMatrix(
        matrix=cell_by_bin,
        barcodes=barcodes,
        bins=bins,
    )

def get_expanded_coordinate_mapping(labels: List[str], all_labels: List[str]) -> Dict[int, int]:
    """
    Given a list of row or column labels for one matrix and a list of the union
    of row or column labels for multiple matrices, create a mapping of row/column
    indices from the original matrix to rows/columns of the "union matrix".

    :param labels: Labels for one individual matrix
    :param all_labels: Union of labels for all matrices
    :return: Mapping from integer coordinates in original matrix to "union matrix"

    >>> get_expanded_coordinate_mapping(['A', 'C'], ['A', 'B', 'C'])
    {0: 0, 1: 2}
    """
    all_label_coord_mapping = {l: i for i, l in enumerate(all_labels)}
    return {i: all_label_coord_mapping[l] for i, l in enumerate(labels)}

def expand_matrices_to_common_dims(data: List[CellByBinMatrix]) -> Tuple[List[scipy.sparse.spmatrix], List[str], List[str]]:
    # It's a bit of a waste to store a reference to the same lists and
    # mappings in each element of the resulting list, but this just needs to work
    all_barcodes = sorted_union(*(d.barcodes for d in data))
    barcode_mappings = [get_expanded_coordinate_mapping(d.barcodes, all_barcodes) for d in data]

    all_bins = sorted_union(*(d.bins for d in data))
    bin_mappings = [get_expanded_coordinate_mapping(d.bins, all_bins) for d in data]

    new_matrices: List[scipy.sparse.spmatrix] = []

    for i, d in enumerate(data):
        barcode_mapping = barcode_mappings[i]
        bin_mapping = bin_mappings[i]

        # Already in COO format, but be defensive
        cbb_coo = d.matrix.tocoo()
        new_rows = np.array([barcode_mapping[row] for row in cbb_coo.row])
        new_cols = np.array([bin_mapping[col] for col in cbb_coo.col])
        new_data = np.ones(cbb_coo.nnz)

        coo_tuple = (new_data, (new_rows, new_cols))
        new_matrix = scipy.sparse.coo_matrix(coo_tuple, shape=(len(all_barcodes), len(all_bins)))
        new_matrices.append(new_matrix.tocsr())

    return new_matrices, all_barcodes, all_bins

def compare_matrices(m1: scipy.sparse.spmatrix, m2: scipy.sparse.spmatrix) -> float:
    """
    :param m1: Sparse binary cell by bin matrix
    :param m2: Sparse binary cell by bin matrix
    :return: Proportion of entries that differ between m1 and m2
    """
    assert m1.shape == m2.shape
    diff = (m1 != m2)
    total_size = np.prod(m1.shape)
    return diff.nnz / total_size

def compare_analysis_results(paths: List[Path]):
    data_matrices = [read_pipeline_output(path) for path in paths]
    expanded, all_barcodes, all_bins = expand_matrices_to_common_dims(data_matrices)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('analysis_directory', type=Path, nargs='+')
    args = p.parse_args()

    compare_analysis_results(args.analysis_directory)
