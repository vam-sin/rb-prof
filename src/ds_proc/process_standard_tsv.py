'''
standard processing done for all datasets, same as the liver, script from francesco
'''
import os
import re

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm.auto import tqdm


def make_dataframe(
    ribo_fname: str, data_path: str, df_trans_to_seq, count_norm: str = "mean"
):
    ribo_fpath = os.path.join(data_path, ribo_fname)

    # Import dataset with ribosome data
    df_ribo = pd.read_csv(
        ribo_fpath,
        sep=" ",
        on_bad_lines="warn",
        dtype=dict(gene="category", transcript="category"),
    ).rename(columns={"count": "counts"})

    # Define count normalization function
    if count_norm == "max":
        f_norm = lambda x: x / x.max()
    elif count_norm == "mean":
        f_norm = lambda x: x / x.mean()
    elif count_norm == "sum":
        f_norm = lambda x: x / x.sum()
    else:
        raise ValueError()

    # Create final dataframe
    final_df = (
        df_ribo.merge(df_trans_to_seq).assign(fname=ribo_fname)
        # Filter spurious positions at the end of the sequence
        .query("position_A_site <= n_codons * 3")
        # Compute normalized counts
        .assign(
            norm_counts=lambda df: df.groupby("gene", observed=True).counts.transform(
                f_norm
            )
        )
    )

    return final_df


def make_all_dataframes(
    data_dirpath: str,
    fa_fpath: str,
    max_n_codons: int = 2000,
    count_norm: str = "mean",
):
    # Import FASTA
    data = []
    with open(fa_fpath, mode="r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append([record.id, str(record.seq)])

    # Create transcripts to sequences mapping
    # removes those sequences that have Ns
    df_trans_to_seq = pd.DataFrame(data, columns=["transcript", "sequence"])
    df_trans_to_seq["sequence"] = df_trans_to_seq["sequence"]
    sequence_has_n = df_trans_to_seq.sequence.str.contains("N", regex=False)
    df_trans_to_seq = df_trans_to_seq.loc[~sequence_has_n]
    df_trans_to_seq = df_trans_to_seq.assign(
        n_codons=lambda df: df.sequence.str.len() // 3
    )

    dfs = [
        make_dataframe(
            f,
            df_trans_to_seq=df_trans_to_seq.drop("sequence", axis=1),
            data_path=data_dirpath,
            count_norm=count_norm,
        )
        for f in tqdm(os.listdir(data_dirpath))
        if not f.startswith("ensembl")
    ]

    dfs = pd.concat(dfs)
    for col in ["transcript", "gene", "fname"]:
        dfs[col] = dfs[col].astype("category")

    # Aggregate replicates
    dfs = dfs.groupby(["transcript", "position_A_site"], observed=True)

    dfs = dfs.agg(dict(norm_counts="mean", gene="first")).reset_index()

    dfs = dfs.assign(codon_idx=lambda df: df.position_A_site // 3)

    dfs = dfs.groupby("transcript", observed=True)

    dfs = dfs.agg(
        {
            "norm_counts": lambda x: x.tolist(),
            "codon_idx": lambda x: x.tolist(),
            "gene": "first",
        }
    ).reset_index()
    dfs = dfs.merge(df_trans_to_seq)

    dfs = dfs.assign(
        n_annot=lambda df: df.norm_counts.transform(lambda x: len(x))
        / (df.sequence.str.len() // 3)
    )

    dfs = dfs.assign(perc_annot=lambda df: df.n_annot / df.n_codons)

    # Filter by max sequence lenght
    dfs = dfs.query("n_codons<@max_n_codons")

    return dfs


if __name__ == '__main__': 
    fa_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr/ensembl.cds.fa'
    data_dir_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr/LEU_ILE_VAL/'

    dfs = make_all_dataframes(data_dir_path, fa_path)

    print(dfs)

    dfs.to_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/raw/LEU_ILE_VAL.csv')
