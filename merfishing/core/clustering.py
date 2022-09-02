import pandas as pd
import scanpy as sc
from anndata import AnnData


def qc_before_clustering(
    cell_meta,
    cell_by_gene,
    snmfish_genes=None,
    blank_gene_sum_high=5,
    z_number=4,
    volume_low=30,
    volumn_high=2000,
    transcript_sum_low=5,
    transcript_sum_high=5000,
    tanscripts_per_volume_low=0.05,
    tanscripts_per_volume_high=8,
):
    """
    QC for cells before clustering.

    Parameters
    ----------
    cell_meta :
        cell_meta file with fov, volume and spatial coordinates information
    cell_by_gene :
        cell by gene matrix
    snmfish_genes :
        wether to delete smfish gene in cell_by_gene matrix
    blank_gene_sum_high :
        maximu number of Bank transcripts on ecell can contain,
    z_number :
        minimum z stack a cell should be on
    volume_low :
        minimum cell volume
    volumn_high :
        maximum cell volume
    transcript_sum_low :
        minimum transcripts per cell
    transcript_sum_high :
        maximum transcripts per cell
    tanscripts_per_volume_low :
        minimum ratio for transcriptsto cell volume
    tanscripts_per_volume_high :
        maximum ratio for transcriptsto cell volume

    Returns
    -------
    cell_by_gene and cell_meta
    """

    # filter by blank genes
    blank_genes = []
    for gene in cell_by_gene.columns:
        if gene.startswith("Blank"):
            blank_genes.append(gene)

    tmp = cell_by_gene[blank_genes].copy()
    tmp["sum"] = tmp.sum(axis=1)
    tmp = tmp[tmp["sum"] < blank_gene_sum_high]

    print(f"{tmp.shape[0]} cells after blank gene QC")

    # filter cells by z_number and cell column
    if z_number is not None:
        cell_meta = cell_meta[cell_meta["z"] > z_number]

    cell_meta = cell_meta[(cell_meta["volume"] > volume_low) & (cell_meta["volume"] < volumn_high)]

    shared = list(set(cell_meta.index) & set(tmp.index))
    cell_meta = cell_meta.loc[shared]
    cell_by_gene = cell_by_gene.loc[shared]

    print(f"{cell_meta.shape[0]} cells after z number and cell volume QC")

    # filter blank gene
    non_blank_genes = []
    for gene in cell_by_gene.columns:
        if gene.startswith("Blank") == False:
            non_blank_genes.append(gene)

    cell_by_gene = cell_by_gene[non_blank_genes]

    if snmfish_genes is not None:
        snmfish_genes = ["Fos", "Gad1", "Mbp", "Slc17a7", "Slc1a2", "Snap25"]
        cell_by_gene = cell_by_gene.drop(snmfish_genes, axis=1)

    # filter cells by transcripts sum
    cell_by_gene["sum"] = cell_by_gene.sum(axis=1)
    cell_by_gene = cell_by_gene[
        (cell_by_gene["sum"] > transcript_sum_low) & (cell_by_gene["sum"] < transcript_sum_high)
        ]
    shared = list(set(cell_meta.index) & set(cell_by_gene.index))
    cell_meta = cell_meta.loc[shared]
    cell_by_gene = cell_by_gene.loc[shared]

    print(f"{cell_meta.shape[0]} cells after transcripts sum QC")

    # filter by tanscripts per volume
    cell_by_gene["volume"] = cell_meta["volume"]
    cell_by_gene["t/v"] = cell_by_gene["sum"] / cell_meta["volume"]

    cell_by_gene = cell_by_gene[
        (cell_by_gene["t/v"] > tanscripts_per_volume_low) & (cell_by_gene["t/v"] < tanscripts_per_volume_high)
        ]
    shared = list(set(cell_meta.index) & set(cell_by_gene.index))
    cell_meta = cell_meta.loc[shared]

    print(f"{cell_meta.shape[0]} cells after transcripts/v QC")

    cell_by_gene = cell_by_gene.drop(["sum", "volume", "t/v"], axis=1)

    cell_by_gene = cell_by_gene.sort_index()
    cell_meta = cell_meta.sort_index()

    print(f"{cell_meta.shape[0]} of cells remained")

    return cell_by_gene, cell_meta


def get_adata(cell_by_gene, cell_meta, n_neighbors=10, n_pcs=20):
    """
    this will generate the adata for clustering

    Parameters
    ----------
    cell_by_gene :
        cell by gene matrix
    cell_meta :
        cell meta frame
    n_neighbors :
        The size of local neighborhood (in terms of number of neighboring data points) used for manifold approximation.
    n_pcs :
        Use this many PCs. Defaults to 20.

    Returns
    -------
    adata
    """
    assert cell_by_gene.shape[0] == cell_meta.shape[0]
    counts = cell_by_gene.to_numpy()
    coordinates = cell_meta[["center_x", "center_y"]].to_numpy()
    adata = AnnData(
        counts,
        dtype=counts.dtype,
        obs=pd.DataFrame([], index=cell_by_gene.index),
        var=pd.DataFrame([], index=cell_by_gene.columns),
        obsm={"spatial": coordinates},
    )

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pl.pca_variance_ratio(adata, log=True)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    return adata
