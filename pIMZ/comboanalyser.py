import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import scipy
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from pylab import *
from scipy.optimize import curve_fit
from scipy.stats import zscore
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from collections import Counter,defaultdict
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class ComboAnalyser:
    """
    spec:         3D Matrix for metabolite intesities
    spec_anno:    annotation vector for the 3rd dimension of spec
    adata:        3D Matrix for gene expression
    adata_anno:   annotation vector for 3rd dimension of gene expression
    mapping:      coordinate mapping between adata and spec
    """

    def __init__(
        self,
        spec: np.array,
        spec_anno: np.array,
        adata: np.array,
        adata_anno: np.array,
        mapping: np.array,
        cache_mappings: bool = True
    ):

        self.spec = spec
        self.trans = adata
        self.mapping = mapping
        self.spec_anno = spec_anno
        self.trans_anno = adata_anno

        # self.spec_gaussian = None
        # self.trans_gaussian = None

        self.spec_tic = self.spec.sum(axis=2)
        self.trans_tic = self.trans.sum(axis=2)

        # print("trans_tic shape" , self.trans_tic.shape)

        self.spec_tic_gaussian = None
        self.trans_tic_gaussian = None

        self.mask = self.get_mask()

        self.spec_greater_zero_mask = self.greater_zero(self.spec)
        self.trans_greater_zero_mask = self.greater_zero(self.trans)

        ### max number of pixel that show metabolite / gene expression
        self.spec_max_pixel = self.greater_zero(self.spec_tic).sum(axis=0).sum(axis=0)
        self.trans_max_pixel = self.greater_zero(self.trans_tic).sum(axis=0).sum(axis=0)

        ### number of pixel for each mass annotation / gene that are > 0
        self.spec_pixel = self.greater_zero(self.spec).sum(axis=0).sum(axis=0)
        self.trans_pixel = self.greater_zero(self.trans).sum(axis=0).sum(axis=0)

        ### mean expression of every mass annotation / gene
        self.spec_mean_expression = self.compute_mean_expression(self.spec)
        self.trans_mean_expression = self.compute_mean_expression(self.trans)

        self.spec_cluster = None
        self.trans_cluster = None

        self.spec_clustered_intensities = None
        self.trans_clustered_intensities = None

        self.spec_cluster_list = None
        self.trans_cluster_list = None

        self.cache_mappings = cache_mappings
        if self.cache_mappings:
            self.mapped_data = {}
        else:
            self.mapped_data = None
        

    def spatially_relevant_features(self, where="both", min_expr_count=1000, min_conn_count=250):

        if where == "both":
            tx_features = self._get_spatially_relevant_features(self.trans_anno, self.trans, min_expr_count=min_expr_count, min_conn_count=min_conn_count)
            px_features = self._get_spatially_relevant_features(self.spec_anno, self.spec, min_expr_count=min_expr_count, min_conn_count=min_conn_count)
            return tx_features, px_features
        
        elif where == "trans":
            tx_features = self._get_spatially_relevant_features(self.trans_anno, self.trans, min_expr_count=min_expr_count, min_conn_count=min_conn_count)
            return tx_features
        
        elif where == "spec":
            px_features = self._get_spatially_relevant_features(self.spec_anno, self.spec, min_expr_count=min_expr_count, min_conn_count=min_conn_count)
            return px_features

        print("where must be 'both', 'trans' or 'spec'")

    def _get_spatially_relevant_features(self, expr_annot, expr, min_expr_count=1000, min_conn_count=250):
        relevant_genes = {}
        for i, gene in tqdm(enumerate(expr_annot), total=len(expr_annot)):
            image = expr[:, :, i]

            try:
                thresholds = threshold_multiotsu(image, classes=2)
                regions = np.digitize(image, bins=thresholds)

                exprcount = min( [(regions == 0).flatten().sum(), (regions == 1).flatten().sum()])

                if exprcount < min_expr_count:
                    continue

                lregions = label(regions)
                conncount = Counter(lregions.flatten()).most_common(10)[1][1]

                if conncount < min_conn_count:
                    continue

                relevant_genes[gene] = regions
            except:
                continue

            
        print(len(relevant_genes))
        return relevant_genes


    """
        checks for all cells in the 3D matrix wether the value ist greater 0 and returns a bool matrix of the same size
    """

    def greater_zero(self, matrix):
        return matrix > 0

    def compute_mean_expression(self, matrix):
        means = np.zeros(matrix.shape[2])
        for i in range(matrix.shape[2]):
            elements = matrix[:, :, i][matrix[:, :, i] > 0]
            if elements.size > 0:
                means[i] = np.mean(elements)

        return means

    """
        applies the greater zero mask on the underlying 3D matrix
        --> resets 0s after applying gaussian filter (because these values are something like 0.00001 
            and impair regression results)
    """

    def apply_greater_zero_mask(self, mask, matrix):
        res_matrix = np.zeros_like(matrix)
        res_matrix[mask] = matrix[mask]
        return res_matrix

    def map_coordinates(self, x, y):
        # x,y sind transcriptomic pixels
        # mapping[x, y]
        return round(self.mapping[y, x, 0]), round(self.mapping[y, x, 1])

    def apply_gaussian_filter_3d(self, matrix_3d, sigma):
        """
        Applies a 2D Gaussian filter to each 2D slice in a 3D matrix along the third dimension.

        Parameters:
        matrix_3d (numpy.ndarray): 3D numpy array.
        sigma (float): Standard deviation for Gaussian kernel.

        Returns:
        numpy.ndarray: 3D numpy array with filtered 2D slices.
        """
        # Get the shape of the input matrix
        depth = matrix_3d.shape[2]

        # Initialize an empty array to store the filtered results
        filtered_matrix_3d = np.zeros_like(matrix_3d)

        # Iterate over each 2D slice in the 3D matrix
        for i in tqdm(range(depth)):
            filtered_matrix_3d[:, :, i] = gaussian_filter(
                matrix_3d[:, :, i], sigma=sigma
            )

        return filtered_matrix_3d

    """
        Calls apply_gaussian_filter_3d() for spec and trans matrices 
        --> replaces spec and trans with gaussian filtered matrices
    """

    def gaussian_filter_on_data(self, sigma=1.0, greater_zero_mask=True):
        print(
            f"sigma (std) = {sigma} , trunced = 4.0 (default, fix) , radius = {round(4.0 * sigma)}"
        )
        print("gaussian filter on spatial metabolomic data ...")
        self.spec = self.apply_gaussian_filter_3d(self.spec, sigma=sigma)
        if greater_zero_mask:
            self.spec = self.apply_greater_zero_mask(
                self.spec_greater_zero_mask, self.spec
            )
        self.spec_tic_gaussian = self.spec.sum(axis=2)

        print("\ngaussian filter on spatial transcriptomic data ...")
        self.trans = self.apply_gaussian_filter_3d(self.trans, sigma=sigma)
        if greater_zero_mask:
            self.trans = self.apply_greater_zero_mask(
                self.trans_greater_zero_mask, self.trans
            )
        self.trans_tic_gaussian = self.trans.sum(axis=2)

        self.spec_mean_expression = self.compute_mean_expression(self.spec)
        self.trans_mean_expression = self.compute_mean_expression(self.trans)

    """
        sumed up intensities for spec or trans are plotted
    """

    def plot_tics(self, obj="spec", gaussian_bool=False):
        title = f"TICs {obj}"

        if (obj == "spec") and (not gaussian_bool):
            data = self.spec_tic
        elif (obj == "trans") and (not gaussian_bool):
            data = self.trans_tic
        elif (obj == "spec") and gaussian_bool and not (self.spec_tic_gaussian is None):
            data = self.spec_tic_gaussian
            title += " gaussian filter"
        elif (
            (obj == "trans") and gaussian_bool and not (self.trans_tic_gaussian is None)
        ):
            data = self.trans_tic_gaussian
            title += " gaussian filter"
        else:
            return

        plt.imshow(data, cmap="viridis", interpolation="nearest")
        plt.colorbar()  # Show color scale
        plt.title(title)
        plt.show()

    """
        the difference before and after gaussian filtering is plotted
    """

    def plot_tic_difference(self, obj="spec"):
        if not self.spec_tic_gaussian is None:
            if obj == "spec":
                data = self.spec_tic - self.spec_tic_gaussian
            elif obj == "trans":
                data = self.trans_tic - self.trans_tic_gaussian

            new_column = np.full((data.shape[0], 1), -abs(np.max(data)))
            data = np.hstack((new_column, data))
            new_column = np.full((data.shape[0], 1), abs(np.max(data)))
            data = np.hstack((data, new_column))

            plt.imshow(data, cmap="seismic", interpolation="nearest")
            plt.colorbar()  # Show color scale
            plt.title(f"TIC difference {obj} gaussian filter")
            plt.show()

    def get_prot_location(self, x, y):
        # x,y sind transcriptomic pixels
        # mapping[x, y]
        return None  # mapping[x][y][0], mapping[x][y][1]

    def get_prot_location_kernel(self):
        pass

    def get_transcriptomics_shape(self):
        # print(self.trans.shape)
        return (self.trans.shape[0], self.trans.shape[1])  # self.trans.shape#

    # map index to gene / annotation name and vice versa
    def get_tx_index_from_feature(self, tx_feature):
        return int(np.where(self.trans_anno == tx_feature)[0][0])

    def get_px_index_from_feature(self, px_feature):
        return int(np.where(self.spec_anno == px_feature)[0][0])

    def get_feature_name_from_tx_index(self, tx_index):
        return self.trans_anno[tx_index]

    def get_feature_name_from_px_index(self, px_index):
        return self.spec_anno[px_index]


    def transform_feature(self, X):

        sx, sy, _ = self.spec.shape
        h, w, _ = self.trans.shape

        # Extract coordinates from the mapping array
        cint = self.mapping.astype(int).copy()

        cintv = cint.reshape(-1, 2)

        cintv[cintv[:, 0] >= h, 0] = h-1
        cintv[cintv[:, 1] >= w, 1] = w-1

        resv = X[cintv[:,1], cintv[:,0]].copy()
        resv = resv.reshape(sx,sy)
        return resv


    def get_feature_distributions(self, si, ti):
        s, t = self.get_features(si, ti)
        return s.flatten(), t.flatten()

    def get_features(self, si, ti):

        s = self.spec[:, :, si]
        
        if self.cache_mappings and ti in self.mapped_data:
            t = self.mapped_data[ti]
        else:
            t = self.trans[:, :, ti]
            t = self.transform_feature(t)
            if self.cache_mappings:
                self.mapped_data[ti] = t

        return s, t

    def get_feature_distributions_per_cluster(self, pi, ti):
        t = self.trans_clustered_intensities[:, ti]
        p = self.spec_clustered_intensities[:, pi]

        return p.flatten(), t.flatten()

    def get_feature_distributions_within_cluster(self, pi, ti, p_cluster, t_cluster):

        t = self.trans[:, :, ti]
        t2 = np.zeros_like(t)
        t2[self.trans_cluster == t_cluster] = t[self.trans_cluster == t_cluster]

        p = self.spec[:, :, pi]
        p2 = np.zeros_like(p)
        p2[self.spec_cluster == p_cluster] = p[self.spec_cluster == p_cluster]

        t2 = self.transform_feature(t2)

        return p2.flatten(), t2.flatten()

    """
        computes the mask
            -->  pixel with 0s in either transcriptomic or metabolomic data are not considered
    """

    def get_mask(self):
        t = self.transform_feature(self.trans_tic)
        p = self.spec_tic

        print(f"t shape: {t.shape} , p shape: {p.shape}")

        return (t > 0) & (p > 0)

    def plot_mask(self):
        plt.imshow(self.mask, cmap="Greys", interpolation="nearest")
        plt.colorbar()  # Show color scale
        plt.title("mask")
        plt.show()

    """
        applies mask to flat transcriptomic and metabolomic data
    """

    def mask_on(self, p_flat, t_flat):
        # bool_array = ( p_flat > 0 ) & ( t_flat > 0 )
        # return p_flat[bool_array] , t_flat[bool_array]
        return p_flat[self.mask.flatten()], t_flat[self.mask.flatten()]

    def remove_zeros(self, p_flat, t_flat):
        bool_array = (p_flat > 0) & (t_flat > 0)
        return p_flat[bool_array], t_flat[bool_array]


    def regression_custom(self, pxVec, txVec, feature_px, feature_tx, plot, series=None):
        method = "regression"

        if len(txVec) == 0 or min(txVec) == max(txVec):
            slope = 0
            intercept = np.mean(pxVec)
            r_value = 0

        else:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                txVec, pxVec
            )  # (x, y)

        m = slope  # coefs[0]
        b = intercept  # coefs[1]
        y_pred = m * txVec + b
        residuals = pxVec - y_pred
        if len(txVec) > 0:
            mae = mean_absolute_error(pxVec, y_pred)
            rms = sqrt(mean_squared_error(pxVec, y_pred))
        else:
            mae = np.nan
            rms = np.nan

        standard_error_of_mean = scipy.stats.sem(residuals)

        if plot:
            self.scatter_hist(
                txVec,
                pxVec,
                title=f"{self.get_feature_name_from_px_index(feature_px)}  vs.  {self.get_feature_name_from_tx_index(feature_tx)}\n{method}\nm = {round(m, 5)} ,   t = {round(b, 5)}",
            )

        mean_expression_spec = self.spec_mean_expression[
            feature_px
        ]  # np.mean(temp[ temp > 0 ] )
        pixel_spec = self.spec_pixel[feature_px]  # temp.sum(axis = 0).sum(axis = 0)

        mean_expression_trans = self.trans_mean_expression[
            feature_tx
        ]  # np.mean(temp[ temp > 0 ] )
        pixel_trans = self.trans_pixel[feature_tx]  # temp.sum(axis = 0).sum(axis = 0)

        frac_pixel_trans = pixel_trans / self.trans_max_pixel
        frac_pixel_spec = pixel_spec / self.spec_max_pixel

        if series is None:
            series = pd.Series()

        series["slope"] = slope
        series["intercept"] = intercept
        series["r_value"] = r_value
        series["standard_error_of_mean"] = standard_error_of_mean
        series["mae"] = mae
        series["rms"] = rms
        series["mean_expression_spec"] = mean_expression_spec
        series["mean_expression_trans"] = mean_expression_trans
        series["txVec_len"] = len(txVec)
        series["pixel_spec"] = pixel_spec
        series["pixel_trans"] = pixel_trans
        series["frac_pixel_spec"] = frac_pixel_spec
        series["frac_pixel_trans"] = frac_pixel_trans

        return series

    def correlation_custom(self, pxVec, txVec, feature_px, feature_tx, plot, series=None):
        corr = np.corrcoef(txVec, pxVec)
        if plot:
            self.plot_correlation(
                txVec,
                pxVec,
                self.get_feature_name_from_tx_index(feature_tx),
                self.get_feature_name_from_px_index(feature_px),
                round(corr[0, 1], 5),
            )

        
        mean_expression_spec = self.spec_mean_expression[feature_px]
        pixel_spec = self.spec_pixel[feature_px]

        mean_expression_trans = self.trans_mean_expression[feature_tx]
        pixel_trans = self.trans_pixel[feature_tx]

        frac_pixel_trans = pixel_trans / self.trans_max_pixel
        frac_pixel_spec = pixel_spec / self.spec_max_pixel


        if series is None:
            series = pd.Series()

        pixel = len(txVec)
        series["corr_coef"] = corr[0, 1]
        series["mean_expression_spec"] = mean_expression_spec
        series["mean_expression_trans"] = mean_expression_trans
        series["pixel_spec"] = pixel_spec
        series["pixel_trans"] = pixel_trans
        series["frac_pixel_spec"] = frac_pixel_spec
        series["frac_pixel_trans"] = frac_pixel_trans
        series["pixel"] = pixel

        return series
    

    def exponential_fit_custom(self, pxVec, txVec, feature_px, feature_tx, plot, series=None):
        method = "exponential"
        if len(txVec) >= 10:
            initial_guess = [np.max(pxVec), np.median(txVec), 0.001, np.min(pxVec)]
            popt, pcov = curve_fit(
                self.exponential_func,
                txVec,
                pxVec,
                p0=initial_guess,
                # p0 = [10,100,0.0001,1] ,
                maxfev=1000000,
            )  #
            a, b, c, d = popt
            pxVec_predict = self.exponential_func(txVec, a, b, c, d)
            r_square = r2_score(pxVec, pxVec_predict)
            rmse = sqrt(mean_squared_error(pxVec, pxVec_predict))

        else:
            a, b, c, d = 0, 0, 0, 0
            r_square = 0
            rmse = 0

        pixel = len(txVec)

        mean_expression_spec = self.spec_mean_expression[feature_px]
        pixel_spec = self.spec_pixel[feature_px]

        mean_expression_trans = self.trans_mean_expression[feature_tx]
        pixel_trans = self.trans_pixel[feature_tx]

        frac_pixel_trans = pixel_trans / self.trans_max_pixel
        frac_pixel_spec = pixel_spec / self.spec_max_pixel

        if plot and pixel >= 10:
            self.scatter_hist_exponential(
                txVec,
                pxVec,
                a,
                b,
                c,
                d,
                f"{self.get_feature_name_from_px_index(feature_px)}  vs.  {self.get_feature_name_from_tx_index(feature_tx)}\n{method}\nR² = {round(r_square , 3)}  RMSE = {round(rmse , 3)}",
            )


        if series is None:
            series = pd.Series()

        series["exponential_a"] = a
        series["exponential_b"] = b
        series["exponential_c"] = c
        series["exponential_d"] = d

        series["r_square"] = r_square
        series["rmse"] = rmse
        series["mean_expression_spec"] = mean_expression_spec
        series["mean_expression_trans"] = mean_expression_trans
        series["pixel_spec"] = pixel_spec
        series["pixel_trans"] = pixel_trans
        series["frac_pixel_spec"] = frac_pixel_spec
        series["frac_pixel_trans"] = frac_pixel_trans
        series["pixel"] = pixel

        return series
    
    def jaccard_custom(self, pxExpr, txExpr, feature_px, feature_tx, plot, series=None):
        method = "jaccard"

        txThresholds = threshold_multiotsu(txExpr, classes=2)
        txRegions = np.digitize(txExpr, bins=txThresholds)

        pxThresholds = threshold_multiotsu(pxExpr, classes=2)
        pxRegions = np.digitize(pxExpr, bins=pxThresholds)

        pxCoords = set([tuple([x,y]) for x,y in zip(*np.where(pxRegions==1))])
        txCoords = set([tuple([x,y]) for x,y in zip(*np.where(txRegions==1))])

        coordIntersect = pxCoords.intersection(txCoords)
        coordUnion = pxCoords.union(txCoords)

        jaccardIndex = len(coordIntersect) / len(coordUnion)


        mean_expression_spec = self.spec_mean_expression[feature_px]
        pixel_spec = self.spec_pixel[feature_px]

        mean_expression_trans = self.trans_mean_expression[feature_tx]
        pixel_trans = self.trans_pixel[feature_tx]

        frac_pixel_trans = pixel_trans / self.trans_max_pixel
        frac_pixel_spec = pixel_spec / self.spec_max_pixel

        if series is None:
            series = pd.Series()

        series["sx_coords"] = len(pxCoords)
        series["tx_oords"] = len(txCoords)

        series["jaccard"] = jaccardIndex

        series["jaccard_intersect"] = len(coordIntersect)
        series["jaccard_union"] = len(coordUnion)

        series["mean_expression_spec"] = mean_expression_spec
        series["mean_expression_trans"] = mean_expression_trans

        series["pixel_spec"] = pixel_spec
        series["pixel_trans"] = pixel_trans

        series["frac_pixel_spec"] = frac_pixel_spec
        series["frac_pixel_trans"] = frac_pixel_trans

        return series


    

    def scatter_hist_exponential(self, x, y, a, b, c, d, title=""):
        # Define the figure and the grid
        fig = plt.figure(figsize=(10, 10))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

        # Main scatter plot with regression line
        ax_main = fig.add_subplot(grid[1:4, 0:3])
        ax_main.scatter(x=x, y=y, alpha=0.5)
        ax_main.set_xlabel("gene expression level")
        ax_main.set_ylabel("metabolite intesity")

        y_pred = self.exponential_func(x, a, b, c, d)

        ax_main.plot(x, y_pred, color="black")

        # Top histogram
        ax_top = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
        ax_top.hist(x, bins=40, color="gray", alpha=0.7)
        ax_top.set_ylabel("Count")
        plt.setp(
            ax_top.get_xticklabels(), visible=False
        )  # Hide x labels for top histogram

        # Right histogram
        ax_right = fig.add_subplot(grid[1:4, 3], sharey=ax_main)
        ax_right.hist(y, bins=40, orientation="horizontal", color="gray", alpha=0.7)
        ax_right.set_xlabel("Count")
        plt.setp(
            ax_right.get_yticklabels(), visible=False
        )  # Hide y labels for right histogram
        plt.suptitle(title)

        plt.show()

    def exponential_func(self, x, a, b, c, d):
        return a * np.exp(-c * (x - b)) + d



    def scatter_hist(self, x, y, title=""):
        # Define the figure and the grid
        fig = plt.figure(figsize=(10, 10))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

        # Main scatter plot with regression line
        ax_main = fig.add_subplot(grid[1:4, 0:3])
        sns.regplot(x=x, y=y, ax=ax_main, scatter_kws={"alpha": 0.5})
        ax_main.set_xlabel("gene expression level")
        ax_main.set_ylabel("metabolite intesity")

        # Top histogram
        ax_top = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
        ax_top.hist(x, bins=40, color="gray", alpha=0.7)
        ax_top.set_ylabel("Count")
        plt.setp(
            ax_top.get_xticklabels(), visible=False
        )  # Hide x labels for top histogram

        # Right histogram
        ax_right = fig.add_subplot(grid[1:4, 3], sharey=ax_main)
        ax_right.hist(y, bins=40, orientation="horizontal", color="gray", alpha=0.7)
        ax_right.set_xlabel("Count")
        plt.setp(
            ax_right.get_yticklabels(), visible=False
        )  # Hide y labels for right histogram
        plt.suptitle(title)

        plt.show()

    def plot_correlation(self, txVec, pxVec, tx_name, px_name, corr_coef):
        # Generate x-axis values from 1 to len(x)
        x_axis = list(range(1, len(txVec) + 1))

        # Normalize the vectors to 100%
        tx_normalized = [val / max(txVec) * 100 for val in txVec]
        px_normalized = [val / max(pxVec) * 100 for val in pxVec]

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))

        # First subplot
        ax1.plot(x_axis, txVec, label=str(tx_name), alpha=0.5)
        ax1.plot(x_axis, pxVec, label=str(px_name), alpha=0.5)
        ax1.set_title(f"Correlation (corr_coef = {corr_coef})")
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Intensities")
        ax1.legend()

        # Second subplot
        ax2.plot(x_axis, tx_normalized, label=str(tx_name), alpha=0.5)
        ax2.plot(x_axis, px_normalized, label=str(px_name), alpha=0.5)
        ax2.set_title("normalized correlation")
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Intensities (%)")
        ax2.legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()


    def one_gene_vs_all(
        self,
        gene,
        trans_cluster=None,
        spec_cluster=None,
        methods=["regression"],
        analysis="pixel",
        remove_zeros=True,
        verbose=False
        ):

        for method in methods:
            if method not in ["regression", "correlation", "exponential", "jaccard"]:
                raise ValueError(
                    f"Method {method} is not supported. Choose from 'regression', 'correlation', 'exponential', or 'jaccard'."
                )

        if analysis not in ["pixel", "per_cluster", "within_cluster"]:
            raise ValueError(
                f"Analysis {analysis} is not supported. Choose from 'pixel', 'per_cluster', or 'within_cluster'."
            )

        createVector = (len(set(["regression", "correlation", "exponential"]).intersection(methods)) > 0)

        if analysis == "per_cluster":
            if self.trans_clustered_intensities is None or self.spec_clustered_intensities is None:
                self.group_data_for_clustering()

        elif analysis == "within_cluster":
            if spec_cluster is None or trans_cluster is None:
                raise ValueError(
                    "For 'within_cluster' analysis, both spec_cluster and trans_cluster must be provided."
                )

        resultsByMethod = defaultdict(list)

        feature_tx_id = self.get_tx_index_from_feature(gene)

        for feature_sx_id in tqdm(range(self.spec.shape[2])):

            feature_sx = self.get_feature_name_from_px_index(feature_sx_id)

            if verbose:
                print("get feature distribution ...")
            pxMap, txMap = self.get_features(feature_sx_id, feature_tx_id)

            if createVector:

                if analysis == "pixel":
                    pxVec, txVec = pxMap.flatten(), txMap.flatten()
                elif analysis == "per_cluster":
                    pxVec, txVec = self.get_feature_distributions_per_cluster(
                        feature_sx_id, feature_tx_id
                    )
                elif analysis == "within_cluster":
                    pxVec, txVec = self.get_feature_distributions_within_cluster(
                        feature_sx_id, feature_tx_id, spec_cluster, trans_cluster
                    )

                if remove_zeros:
                    pxVec, txVec = self.remove_zeros(pxVec, txVec)
                if verbose:
                    print(f"txVec & pxVec has length {len(txVec)} / {len(pxVec)}")

            series = pd.Series([feature_sx_id, feature_sx, feature_tx_id, gene, analysis], index=["feature_sx_id", "feature_sx", "feature_tx_id", "feature_tx", "analysis"])

            for method in methods:      

                # regression
                if method == "regression":
                    if verbose:
                        print("compute linear regression ...")
                    
                    resultsByMethod[method].append(self.regression_custom(
                        pxVec=pxVec,
                        txVec=txVec,
                        feature_px=feature_sx_id,
                        feature_tx=feature_tx_id,
                        plot=False,
                        series=series.copy()
                    ))

                # correlation
                elif method == "correlation":
                    if verbose:
                        print("compute correlation ...")
                    resultsByMethod[method].append(self.correlation_custom(
                        pxVec=pxVec,
                        txVec=txVec,
                        feature_px=feature_sx_id,
                        feature_tx=feature_tx_id,
                        plot=False,
                        series=series.copy()
                    ))

                elif method == "exponential":

                    if verbose:
                        print("compute exponential fit ...")
                    resultsByMethod[method].append(self.exponential_fit_custom(
                        pxVec=pxVec,
                        txVec=txVec,
                        feature_px=feature_sx_id,
                        feature_tx=feature_tx_id,
                        plot=False,
                        series=series.copy()
                    ))
                
                elif method == "jaccard":

                    if verbose:
                        print("compute jaccard methdod ...")

                    resultsByMethod[method].append(self.jaccard_custom(
                        pxExpr=pxMap,
                        txExpr=txMap,
                        feature_px=feature_sx_id,
                        feature_tx=feature_tx_id,
                        plot=False,
                        series=series.copy()
                    ))
            

        final_dfs = {}
        for method in resultsByMethod:
            final_dfs[method] = pd.concat(resultsByMethod[method], axis=1).T

        return final_dfs

    def top_n_intesity_metabolites(self, n):
        tic = self.spec.sum(axis=0).sum(axis=0)

        print(tic.shape)
        print(tic[:10])

        sorted_indices = np.argsort(tic)
        sorted_indices_descending = sorted_indices[::-1]
        top_n_indices = sorted_indices_descending[:n]
        return top_n_indices

    def plot_gene_metabolite_pair(self, tx_index, px_index, exponential=True):
        return self.plot_trans_spec_pair(tx_index, px_index, exponential=exponential)


    def plot_trans_spec_pair(self, tx_index, px_index, analysis="pixel", spec_cluster=None, trans_cluster=None, exponential=True):



        if analysis == "per_cluster":
            if self.trans_clustered_intensities is None or self.spec_clustered_intensities is None:
                self.group_data_for_clustering()
                
        elif analysis == "within_cluster":
            if spec_cluster is None or trans_cluster is None:
                raise ValueError(
                    "For 'within_cluster' analysis, both spec_cluster and trans_cluster must be provided."
                )



        pxVec, txVec = self.get_feature_distributions(px_index, tx_index)
        pxVec, txVec = self.remove_zeros(pxVec, txVec)

        pxMap, txMap = self.get_features(px_index, tx_index)       

        if analysis == "pixel":
            pxVec, txVec = pxMap.flatten(), txMap.flatten()
        elif analysis == "per_cluster":
            pxVec, txVec = self.get_feature_distributions_per_cluster(
                px_index, tx_index
            )
        elif analysis == "within_cluster":
            pxVec, txVec = self.get_feature_distributions_within_cluster(
                px_index, tx_index, spec_cluster, trans_cluster
            )

        pxVec, txVec = self.remove_zeros(pxVec, txVec)
        txMap_original = self.trans[:, :, tx_index]




        fig = plt.figure(figsize=(25, 20))
        #gs = fig.add_gridspec(
        #    4, 7, width_ratios=[6, 1, 0.25, 6, 6, 6, 6], height_ratios=[1, 2, 2, 1]
        #)


        gs = fig.add_gridspec(4, 2, width_ratios=[1, 4])

        gs0 = gs[0].subgridspec(4, 3, wspace=0.1, hspace=0.1, width_ratios=[4, 0.5, 0.5], height_ratios=[1, 2, 2, 1])
        gs1 = gs[1].subgridspec(5, 4, wspace=0.5, height_ratios=[1, 2, 1, 2, 1])


        se_reg = self.regression_custom(pxVec, txVec, px_index, tx_index, plot=False)

        ax_hexa = fig.add_subplot(gs0[1:4, 0])

        hexa = ax_hexa.hexbin(x=txVec, y=pxVec, gridsize=30, cmap="Blues")
        #cbar0 = fig.colorbar(hexa, ax=ax_hexa)
        #cbar0.set_label("counts")
        # ax_hexa.colorbar(label='counts')
        ax_hexa.set_xlabel("gene expression")
        ax_hexa.set_ylabel("metabolite intensity")
        # ax_hexa.title('Hexbin Plot with Regression Line')
        reg_y = se_reg["slope"] * txVec + se_reg["intercept"]
        # Overlay the regression line
        ax_hexa.plot(txVec, reg_y, color="black", linewidth=1, label="lin")

        # TODO: sorry have to remove exponential fit for cell type analysis
        if exponential:
            se_exp = self.exponential_fit_custom(pxVec, txVec, px_index, tx_index, plot=False)

            exp_y = self.exponential_func(txVec, se_exp["exponential_a"], se_exp["exponential_b"], se_exp["exponential_c"], se_exp["exponential_d"])
            sorted_indices = np.argsort(txVec)
            ax_hexa.plot(
                txVec[sorted_indices],
                exp_y[sorted_indices],
                color="grey",
                linewidth=1,
                label="exp",
            )

        ax_hexa.legend()

        # Histograms
        ax_histx = fig.add_subplot(gs0[0, 0], sharex=ax_hexa)
        ax_histy = fig.add_subplot(gs0[1:4, 1], sharey=ax_hexa)

        cbar_ax = fig.add_subplot(gs0[1:4, 2])
        fig.colorbar(hexa, cax=cbar_ax)
        cbar_ax.set_label("counts")

        ax_histx.hist(txVec, bins=40, edgecolor="black")

        if exponential:
            ax_histx.set_title(
                f"regression (R² = {round(se_reg["r_value"] * se_reg["r_value"] , 3)} , RMSE = {round(se_reg["rms"] , 3)})\nexponential (R² = {round(se_exp["r_square"] , 3)} , RMSE = {round(se_exp["rmse"] , 3)})\npixel = {se_exp["pixel"]}"
            )
        else:
            ax_histx.set_title(
                f"regression (R² = {round(se_reg["r_value"] * se_reg["r_value"] , 3)} , RMSE = {round(se_reg["rms"] , 3)})\npixel = {se_reg["pixel"]}"
            )

        ax_histy.hist(pxVec, bins=40, orientation="horizontal", edgecolor="black")

        # Remove ticks on the histogram plots
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)

        #print("Figure 1 done")


        # Second plot
        tx_name = self.get_feature_name_from_tx_index(tx_index)
        px_name = self.get_feature_name_from_px_index(px_index)

        se_corr = self.correlation_custom(pxVec, txVec, px_index, tx_index, plot=False)

        x_axis = list(range(1, len(txVec) + 1))
        ax2 = fig.add_subplot(gs1[0:2, 0])
        ax2.plot(x_axis, txVec, label=str(tx_name), alpha=0.5)
        ax2.plot(x_axis, pxVec, label=str(px_name), alpha=0.5)
        ax2.set_title(f"Correlation (corr_coef = {se_corr["corr_coef"]})\npixel = {se_corr["pixel_spec"]}")
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Intensities")
        ax2.legend()

        #print("Figure 2_1 done")

        # Normalize the vectors to 100%
        tx_normalized = (txVec-min(txVec))
        tx_normalized = tx_normalized / max(tx_normalized)

        px_normalized = (pxVec-min(pxVec))
        px_normalized = px_normalized / max(px_normalized)


        ax2_2 = fig.add_subplot(gs1[3:5, 0])
        ax2_2.plot(x_axis, tx_normalized, label=str(tx_name), alpha=0.5)
        ax2_2.plot(x_axis, px_normalized, label=str(px_name), alpha=0.5)
        ax2_2.set_title("normalized correlation")
        ax2_2.set_xlabel("Index")
        ax2_2.set_ylabel("Intensities (%)")
        ax2_2.legend()
        # Adjust layout
        plt.tight_layout()

        #print("Figure 2_2 done")

        # Third plot
        ax3 = fig.add_subplot(gs1[:, 1])
        im = ax3.imshow(txMap_original, cmap="viridis", interpolation="nearest")

        ax3.set_title(
            f"gene expression\nmean expression = {round( se_corr["mean_expression_trans"] , 3)}\ntranscript name = {tx_name}\ntranscriptomic pixel = {se_corr["pixel_trans"]}"
        )
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label("Intensity")  # Label for the colorbar

        #print("Figure 3 done")


        # Fourth plot
        ax4 = fig.add_subplot(gs1[:, 2])
        im = ax4.imshow(pxMap, cmap="viridis", interpolation="nearest")

        ax4.set_title(
            f"metabolite expression\nmean expression = {round(se_corr["mean_expression_spec"] , 3)}\nmetabolite name = {px_name}\nmetabolite pixel = {se_corr["pixel_spec"]}"
        )
        cbar = fig.colorbar(im, ax=ax4)
        cbar.set_label("Intensity")  # Label for the colorbar

        #print("Figure 4 done")

        # Fifth plot
        ax5 = fig.add_subplot(gs1[:, 3])
        im = ax5.imshow(txMap, cmap="viridis", interpolation="nearest")

        ax5.set_title(
            f"mapped gene expression\nmean expression = {round( se_corr["mean_expression_trans"] , 3)}\ntranscript name = {tx_name}\ntranscriptomic pixel = {se_corr["pixel_trans"]}"
        )
        cbar = fig.colorbar(im, ax=ax5)
        cbar.set_label("Intensity")  # Label for the colorbar

        #print("Figure 5 done")

        return fig

    
    def apply_fraction_filter(self, fraction=1.00):
        print(
            f"only keep the top {fraction * 100} % data points (0 values are omitted)"
        )

        f = 1 - fraction

        spec_threshold = np.copy(self.spec)
        spec_threshold = spec_threshold.flatten()
        spec_threshold = spec_threshold[spec_threshold > 0]
        spec_threshold = np.quantile(spec_threshold, f)
        print(f"metabolite threshold = {spec_threshold}")

        trans_threshold = np.copy(self.trans)
        trans_threshold = trans_threshold.flatten()
        trans_threshold = trans_threshold[trans_threshold > 0]
        trans_threshold = np.quantile(trans_threshold, f)
        print(f"transcriptomic threshold = {trans_threshold}")

        print("apply threshold on spec ...")
        temp = np.zeros_like(self.spec)
        temp_mask = self.spec > spec_threshold
        temp[temp_mask] = self.spec[temp_mask]
        self.spec = np.copy(temp)

        print("apply threshold on trans ...")
        temp = np.zeros_like(self.trans)
        temp_mask = self.trans > trans_threshold
        temp[temp_mask] = self.trans[temp_mask]
        self.trans = np.copy(temp)

    def apply_threshold_filter(self, tx_threshold, px_threshold):
        print(
            f"only keep metabolites with intensity > {px_threshold} and genes with gene expression > {tx_threshold}"
        )

        print("apply threshold on spec ...")
        temp = np.zeros_like(self.spec)
        temp_mask = self.spec > px_threshold
        temp[temp_mask] = self.spec[temp_mask]
        self.spec = np.copy(temp)

        print("apply threshold on trans ...")
        temp = np.zeros_like(self.trans)
        temp_mask = self.trans > tx_threshold
        temp[temp_mask] = self.trans[temp_mask]
        self.trans = np.copy(temp)

    def add_spec_clustering(self, spec_cl):
        print("adding metabolite clustering ...")
        if spec_cl.shape == self.spec_tic.shape:
            self.spec_cluster = spec_cl
            print(
                f"{len(np.unique(spec_cl))} clusters (min: {np.min(spec_cl)} , max: {np.max(spec_cl)})"
            )
            self.spec_cluster_list = np.sort(np.unique(self.spec_cluster))
        else:
            print(
                "shape of the clustering does not match the shape of spatial metabolite data !"
            )

    def add_trans_clustering(self, trans_cl, map_to_spec=False):
        print("adding transcriptomic clustering ...")
        if trans_cl.shape == self.trans_tic.shape:
            
            new_clusters = np.zeros_like(trans_cl)
            for ci, cl in enumerate(np.unique(trans_cl.flatten().tolist()[0])):
                new_clusters[trans_cl == cl] = ci
            self.trans_cluster = new_clusters

            print(
                f"{len(np.unique(self.trans_cluster))} clusters (min: {np.min(self.trans_cluster)} , max: {np.max(self.trans_cluster)})"
            )
            self.trans_cluster_list = np.sort(np.unique(self.trans_cluster))

            if map_to_spec:
                spec_cl = self.transform_feature(self.trans_cluster.copy())
                self.add_spec_clustering(spec_cl)
        else:
            print(
                "shape of the clustering does not match the shape of spatial transcriptomic data !"
            )

    def plot_clustering(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        im1 = ax1.imshow(self.trans_cluster, cmap="viridis")
        ax1.set_title("Gene Expression Clustering")
        cbar1 = fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(self.spec_cluster, cmap="viridis")
        ax2.set_title("Metabolite Clustering")
        cbar2 = fig.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()

    def plot_individual_clusters(self, obj="spec"):
        if obj == "spec":
            clustering = self.spec_cluster
        elif obj == "trans":
            clustering = self.trans_cluster
        else:
            return "obj needs to be either trans or spec !"

        n = len(np.unique(clustering))

        plots_per_row = 5
        # Calculate the number of rows needed
        num_rows = (n + plots_per_row - 1) // plots_per_row

        # Create a figure
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 3 * num_rows))
        fig.suptitle(f"Individual {obj} clusters", fontsize=16)

        i = 0
        for cluster in np.sort(np.unique(clustering)):
            data = clustering == cluster

            # Determine row and column index
            row = i // plots_per_row
            col = i % plots_per_row

            # Get the axis
            ax = axes[row, col] if num_rows > 1 else axes[col]

            # Plot the data
            im = ax.imshow(data, cmap="viridis")
            ax.set_title(f"{cluster}")

            # Remove axes for a cleaner look
            ax.axis("off")

            i += 1

        # Hide any unused subplots
        for j in range(n, num_rows * plots_per_row):
            row = j // plots_per_row
            col = j % plots_per_row
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.axis("off")

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_spec_cluster_expression(self, px):

        spec_temp = self.spec_clustered_intensities[:, px]

        unique_spec_clusters = np.sort(np.unique(self.spec_cluster))
        cluster_value_dict = {}
        for i in range(len(unique_spec_clusters)):
            cluster_value_dict[unique_spec_clusters[i]] = spec_temp[i]
        data = np.zeros_like(self.spec_cluster)
        for i in range(self.spec_cluster.shape[0]):
            for j in range(self.spec_cluster.shape[1]):
                data[i, j] = cluster_value_dict[self.spec_cluster[i, j]]

        plt.imshow(data)
        plt.colorbar()
        plt.title(
            f"cluster-wise aggregated\nmetabolite expression of\n{self.get_feature_name_from_px_index(px)}"
        )
        plt.show()

    def plot_trans_cluster_expression(self, tx):

        trans_temp = self.trans_clustered_intensities[:, tx]

        unique_trans_clusters = np.sort(np.unique(self.trans_cluster))
        cluster_value_dict = {}
        for i in range(len(unique_trans_clusters)):
            cluster_value_dict[unique_trans_clusters[i]] = trans_temp[i]
        data = np.zeros_like(self.trans_cluster)
        for i in range(self.trans_cluster.shape[0]):
            for j in range(self.trans_cluster.shape[1]):
                data[i, j] = cluster_value_dict[self.trans_cluster[i, j]]

        plt.imshow(data)
        plt.colorbar()
        plt.title(
            f"cluster-wise aggregated\ngene expression of\n{self.get_feature_name_from_tx_index(tx)}"
        )
        plt.show()

    def group_data_for_clustering(self, method="sum"):

        ### Metabolites:
        print("group metabolite expression for clusters ...")
        group_data = np.zeros((len(np.unique(self.spec_cluster.flatten().tolist())), self.spec.shape[2]))
        cluster_list = np.sort(np.unique(self.spec_cluster.flatten().tolist()))
        print(cluster_list)
        count = 0
        for cluster in tqdm(cluster_list):
            # cur = self.spec.copy()
            # cur = cur.reshape(-1 , cur.shape[2])
            # cl = self.spec_cluster.copy()
            # cl = cl.flatten()
            # cl = cl == cluster
            # cur = cur[cl , :]
            cur = self.spec[self.spec_cluster == cluster, :]

            if method == "sum":
                group_data[count, :] = cur.sum(axis=0)
            elif method == "mean":
                group_data[count, :] = cur.mean(axis=0)
            elif method == "max":
                group_data[count, :] = np.max(cur, axis=0)
            elif method == "min":
                group_data[count, :] = np.min(cur, axis=0)
            elif method == "median":
                group_data[count, :] = np.median(cur, axis=0)
            elif method == "var":
                group_data[count, :] = np.var(cur, axis=0)
            count += 1
        self.spec_clustered_intensities = group_data

        ### Genes:
        print("group gene expression for clusters ...")
        group_data = np.zeros((len(np.unique(self.trans_cluster.flatten().tolist())), self.trans.shape[2]))
        cluster_list = np.sort(np.unique(self.trans_cluster.flatten().tolist()))
        print(cluster_list)
        count = 0
        for cluster in tqdm(cluster_list):
            cur = self.trans[self.trans_cluster == cluster, :]
            if method == "sum":
                group_data[count, :] = cur.sum(axis=0)
            elif method == "mean":
                group_data[count, :] = cur.mean(axis=0)
            elif method == "max":
                group_data[count, :] = cur.max(axis=0)
            elif method == "min":
                group_data[count, :] = cur.min(axis=0)
            elif method == "median":
                group_data[count, :] = np.median(cur, axis=0)
            elif method == "var":
                group_data[count, :] = np.var(cur, axis=0)
            count += 1
        self.trans_clustered_intensities = group_data



    def z_scale(self, m):
        return (m - np.mean(m)) / np.std(m)  # zscore(matrix)

    def execute(self, Y, X):
        for i in range(X.shape[2]):
            x = X[:, :, j]





class LassoAnalyzer:

    def lasso_1_n(self, y, X, regularization):

        y = y.flatten()

        X = X.reshape(y.shape[0], X.shape[2])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = Lasso(alpha=regularization)
        result = lasso.fit(X_scaled, y)

        coefs = lasso.coef_

        gene_indices = np.where(coefs != 0)[0]

        coefs = coefs[gene_indices]

        idx = np.argsort(coefs * -1)

        coefs = coefs[idx]

        gene_indices = gene_indices[idx]

        return gene_indices, coefs

    def lasso_n_n(self, Y, X, regularization, path, y_labels=None, x_labels=None):
        with open(path, "w") as f:
            for i in range(Y.shape[2]):

                y_label = y_labels[i] if y_labels is not None else ""

                y = Y[:, :, i]

                result_indices, coefs = self.lasso_1_n(y, X, regularization)

                print(
                    f"step {i} of {Y.shape[2]} : {y_label}, found: {result_indices.shape[0]}"
                )

                for j in range(result_indices.shape[0]):
                    x_label = (
                        x_labels[result_indices[j]] if x_labels is not None else ""
                    )
                    f.write(
                        f"{i}\t{y_label}\t{result_indices[j]}\t{x_label}\t{coefs[j]}\n"
                    )

    def execute(self, Y, X, regularization, outpath, y_labels=None, x_labels=None):
        self.lasso_n_n(Y, X, regularization, outpath, y_labels, x_labels)


class Analyser:

    analysers = []

    def __init__(self, analysers):
        self.analysers = analysers

    def execute(Y, X):

        for analyser in analysers:
            analyser.execute(Y, X)
