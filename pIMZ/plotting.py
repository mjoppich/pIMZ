import dabest
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from matplotlib.colors import ColorConverter, to_hex
from matplotlib.path import Path
from matplotlib.cm import ScalarMappable, hsv
import matplotlib.patches as patches

from matplotlib.colors import ColorConverter, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from collections.abc import Sequence
from textwrap import wrap
import scipy.sparse as ssp
from scipy.special import expit

import math


class Plotter():
    dot_shape = "8"
    dot_shape_size = 1.0
    

    @classmethod
    def initialize(cls, figsize=(12,8), dpi=300):

        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.dpi"] = dpi


    @classmethod
    def plot_df_dots(cls, fig, df, xaxis, yaxis, fill, scale, title, cmap="viridis"):

        ax = fig.get_axes()[0]

        scaleFunc = lambda x: (x**2)*120

        scatterMap = ax.scatter(df[xaxis].apply(str), df[yaxis].astype(str), c = df[fill], s=scaleFunc(df[scale]), cmap=cmap)
        plt.xticks(rotation=45,  ha='right')
        plt.colorbar(scatterMap, label=fill)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

        ax.set_title(title, y=1.25)

        #make a legend:
        pws = [np.round(x, 2) for x in np.linspace(0, max(df[scale]), num=5)]
        for pw in pws:
            ax.scatter([], [], s=scaleFunc(pw), c="k",label=str(pw))

        h, l = ax.get_legend_handles_labels()
        plt.legend(h[1:], l[1:], labelspacing=1.2, title=scale, borderpad=0.2, 
                    frameon=True, framealpha=0.6, edgecolor="k", facecolor="w", bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode="expand", ncol=5)



    @classmethod
    def _plot_arrays_grouped(cls, regions, group1, group2, discrete_legend=False, log=False, figsize=(8,6), colorbar_fraction=0.15, title_font_size=12):
        
        colsPerGroup = 3
        
        lgroup1 = list(group1)
        lgroup2 = list(group2)
        
        rowsGroup1 = math.ceil(len(group1)/colsPerGroup)
        rowsGroup2 = math.ceil(len(group2)/colsPerGroup)
        
        numRows = max([rowsGroup1, rowsGroup2])
               
        vrange_min, vrange_max = np.inf,-np.inf
        
        for x in lgroup1+lgroup2:
            
            minval = np.min(regions[x])
            maxval = np.max(regions[x])
            
            vrange_min = min(minval, vrange_min)
            vrange_max = max(maxval, vrange_max)
                   
        fig, axes = plt.subplots(nrows=numRows, ncols=2*colsPerGroup, sharex=False, sharey=False, figsize=figsize)
                
        normalizer=matplotlib.colors.Normalize(vrange_min,vrange_max)
        im=matplotlib.cm.ScalarMappable(norm=normalizer)
        
        usedCoords = []
        
        def set_axis_color(ax, color):
            ax.spines['bottom'].set_color(color)
            ax.spines['top'].set_color(color) 
            ax.spines['right'].set_color(color)
            ax.spines['left'].set_color(color)
        
        groupOffset = 0
        for ci, cname in enumerate(lgroup1):
            
            colindex = ci % colsPerGroup
            rowindex = int(np.floor(ci / colsPerGroup))

            coord = (rowindex, groupOffset+colindex)
            
            ax = axes[coord]
            usedCoords.append(coord)
            
            set_axis_color(ax, "darkred")
            
            cls.plot_array_scatter(regions[cname], ax=ax, discrete_legend=discrete_legend, norm=normalizer)
            ax.set_title(str(cname), fontsize=title_font_size)

        groupOffset = colsPerGroup
        for ci, cname in enumerate(lgroup2):
            
            colindex = ci % colsPerGroup
            rowindex = int(np.floor(ci / colsPerGroup))
            
            coord = (rowindex, groupOffset+colindex)
            
            ax = axes[coord]
            usedCoords.append(coord)
            
            set_axis_color(ax, "darkgreen")
            
            cls.plot_array_scatter(regions[cname], ax=ax, discrete_legend=discrete_legend, norm=normalizer)
            ax.set_title(str(cname), fontsize=title_font_size)

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                
                if not (i,j) in usedCoords:
                    fig.delaxes(axes[(i,j)])


        line=plt.Line2D([0.525, 0.525], [0.05,0.95], color='purple', transform=fig.transFigure)
        fig.add_artist(line)

        fig.subplots_adjust(right=1.0-colorbar_fraction, wspace=0.4, hspace=0.4)
        
        cbar_ax = fig.add_axes([1.0-(colorbar_fraction-0.05), 0.15, (colorbar_fraction-0.05)/2, 0.7], label='Log Intensity')
        fig.colorbar(im, ax=axes.ravel().tolist(), cax=cbar_ax, shrink=0.6)
        cbar_ax.yaxis.set_ticks_position('left')
        
        
        if log:
            cbar_ax.set_xlabel("Log Intensity")
        else:
            cbar_ax.set_xlabel("Intensity")


    @classmethod
    def plot_array_scatter(cls, arr, fig=None, ax=None, discrete_legend=True, norm=None):
        
        shapeSize = (cls.dot_shape_size**2) *0.33

        valid_vals = sorted(np.unique(arr))
        if discrete_legend:
            cmap = plt.cm.get_cmap('viridis', len(valid_vals))
            normArray = np.zeros(arr.shape)
            val_lookup = {}
            positions = []
            for uIdx, uVal in enumerate(valid_vals):
                normArray[arr == uVal] = uIdx
                val_lookup[uIdx] = uVal
                positions.append(uIdx)

        else:
            cmap = plt.cm.get_cmap('viridis')
            
            if norm is None:
                cnorm = matplotlib.colors.Normalize()
                cnorm.autoscale(arr)
            else:
                cnorm = norm
        

        xs = []
        ys = []
        vals = []

        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):

                xs.append(j)
                ys.append(i)
                
                if discrete_legend:
                    val = normArray[i,j]
                    vals.append( val )
                else:
                    val = arr[i,j]
                    vals.append( val )

        if ax is None:
            if fig is None:
                ax = plt.gcf().get_axes()[0]
            else:
                ax = fig.get_axes()[0]
            
        if discrete_legend:
            # calculate the POSITION of the tick labels
            #positions = np.linspace(0, len(valid_vals), len(valid_vals))

            def formatter_func(x, pos):
                'The two args are the value and tick position'
                val = val_lookup[x]
                return val

            formatter = plt.FuncFormatter(formatter_func)

            ax.set_aspect('equal', 'box')
            ax.invert_yaxis()
            heatmap = ax.scatter(xs, ys, c=vals, s=shapeSize, marker=cls.dot_shape, cmap=cmap)
            plt.colorbar(heatmap, ticks=positions, format=formatter, spacing='proportional')

        else:
            ax.set_aspect('equal', 'box')
            ax.invert_yaxis()
            
            heatmap = ax.scatter(xs, ys, c=vals, s=shapeSize, marker=cls.dot_shape, cmap=cmap, norm=cnorm)
            
            if norm is None:
                plt.colorbar(heatmap)
            

    LW = 0.3

    @classmethod
    def chord_diagram(cls, mat, names=None, order=None, width=0.1, pad=2., gap=0.03, delayedArcs=None,
                    chordwidth=0.7, ax=None, colors=None, cmap=None, alpha=0.7,startdegree=90,
                    use_gradient=False, chord_colors=None, show=False, arccolors=None, **kwargs):
        """
        Plot a chord diagram.

        Parameters
        ----------
        mat : square matrix
            Flux data, mat[i, j] is the flux from i to j
        names : list of str, optional (default: no names)
            Names of the nodes that will be displayed (must be ordered as the
            matrix entries).
        order : list, optional (default: order of the matrix entries)
            Order in which the arcs should be placed around the trigonometric
            circle.
        width : float, optional (default: 0.1)
            Width/thickness of the ideogram arc.
        pad : float, optional (default: 2)
            Distance between two neighboring ideogram arcs. Unit: degree.
        gap : float, optional (default: 0)
            Distance between the arc and the beginning of the cord.
        chordwidth : float, optional (default: 0.7)
            Position of the control points for the chords, controlling their shape.
        ax : matplotlib axis, optional (default: new axis)
            Matplotlib axis where the plot should be drawn.
        colors : list, optional (default: from `cmap`)
            List of user defined colors or floats.
        cmap : str or colormap object (default: viridis)
            Colormap that will be used to color the arcs and chords by default.
            See `chord_colors` to use different colors for chords.
        alpha : float in [0, 1], optional (default: 0.7)
            Opacity of the chord diagram.
        use_gradient : bool, optional (default: False)
            Whether a gradient should be use so that chord extremities have the
            same color as the arc they belong to.
        chord_colors : str, or list of colors, optional (default: None)
            Specify color(s) to fill the chords differently from the arcs.
            When the keyword is not used, chord colors default to the colomap given
            by `colors`.
            Possible values for `chord_colors` are:

            * a single color (do not use an RGB tuple, use hex format instead),
            e.g. "red" or "#ff0000"; all chords will have this color
            * a list of colors, e.g. ``["red", "green", "blue"]``, one per node
            (in this case, RGB tuples are accepted as entries to the list).
            Each chord will get its color from its associated source node, or
            from both nodes if `use_gradient` is True.
        show : bool, optional (default: False)
            Whether the plot should be displayed immediately via an automatic call
            to `plt.show()`.
        arccolors : function to retrieve specific colors for arcs
        kwargs : keyword arguments
            Available kwargs are:

            ================  ==================  ===============================
                Name               Type           Purpose and possible values
            ================  ==================  ===============================
            fontcolor         str or list         Color of the names
            fontsize          int                 Size of the font for names
            rotate_names      (list of) bool(s)   Rotate names by 90°
            sort              str                 Either "size" or "distance"
            zero_entry_size   float               Size of zero-weight reciprocal
            ================  ==================  ===============================
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        # copy matrix and set a minimal value for visibility of zero fluxes
        is_sparse = ssp.issparse(mat)

        if is_sparse:
            mat = mat.tocsr(copy=True)
        else:
            mat = np.array(mat, copy=True)

        # mat[i, j]:  i -> j
        num_nodes = mat.shape[0]

        # set entry size for zero entries that have a nonzero reciprocal
        min_deg  = kwargs.get("zero_entry_size", 0.5)
        min_deg *= mat.sum() / (360 - num_nodes*pad)

        if is_sparse:
            nnz = mat.nonzero()

            for i, j in zip(*nnz):
                if mat[j, i] == 0:
                    mat[j, i] = min_deg
        else:
            zeros = np.argwhere(mat == 0)

            for (i, j) in zeros:
                if mat[j, i] != 0:
                    mat[i, j] = min_deg

        # check name rotations
        rotate_names = kwargs.get("rotate_names", False)

        if isinstance(rotate_names, Sequence):
            assert len(rotate_names) == num_nodes, \
                "Wrong number of entries in 'rotate_names'."
        else:
            rotate_names = [rotate_names]*num_nodes

        # check order
        if order is not None:
            mat = mat[order][:, order]

            rotate_names = [rotate_names[i] for i in order]

            if names is not None:
                names = [names[i] for i in order]

        # sum over rows
        x = mat.sum(axis=1).A1 if is_sparse else mat.sum(axis=1)

        # configure colors
        if colors is None:
            colors = np.linspace(0, 1, num_nodes)

        fontcolor = kwargs.get("fontcolor", "k")

        if isinstance(fontcolor, str):
            fontcolor = [fontcolor]*num_nodes
        else:
            assert len(fontcolor) == num_nodes, \
                "One fontcolor per node is required."

        if cmap is None:
            cmap = "viridis"

        if isinstance(colors, (list, tuple, np.ndarray)):
            assert len(colors) == num_nodes, "One color per node is required. {} colors vs {} nodes".format(len(colors), num_nodes)

            # check color type
            first_color = colors[0]

            if isinstance(first_color, (int, float, np.integer)):
                cm = plt.get_cmap(cmap)
                colors = cm(colors)[:, :3]
            else:
                colors = [ColorConverter.to_rgb(c) for c in colors]
        else:
            raise ValueError("`colors` should be a list.")

        if chord_colors is None:
            chord_colors = colors
        else:
            try:
                chord_colors = [ColorConverter.to_rgb(chord_colors)] * num_nodes
            except ValueError:
                assert len(chord_colors) == num_nodes, \
                    "If `chord_colors` is a list of colors, it should include " \
                    "one color per node (here {} colors).".format(num_nodes)

        # find position for each start and end
        y = x / np.sum(x).astype(float) * (360 - pad*len(x))

        pos = {}
        arc = []
        nodePos = []
        rotation = []
        start = startdegree

        # compute all values and optionally apply sort
        for i in range(num_nodes):
            end = start + y[i]
            arc.append((start, end))
            angle = 0.5*(start+end)

            if -30 <= angle <= 180:
                angle -= 90
                rotation.append(False)
            else:
                angle -= 270
                rotation.append(True)

            nodePos.append(
                tuple(cls.polar2xy(1.05, 0.5*(start + end)*np.pi/180.)) + (angle,))

            z = cls._get_normed_line(mat, i, x, start, end, is_sparse)

            # sort chords
            ids = None

            if kwargs.get("sort", "size") == "size":
                ids = np.argsort(z)
            elif kwargs["sort"] == "distance":
                remainder = 0 if num_nodes % 2 else -1

                ids  = list(range(i - int(0.5*num_nodes), i))[::-1]
                ids += [i]
                ids += list(range(i + int(0.5*num_nodes) + remainder, i, -1))

                # put them back into [0, num_nodes[
                ids = np.array(ids)
                ids[ids < 0] += num_nodes
                ids[ids >= num_nodes] -= num_nodes
            else:
                raise ValueError("Invalid `sort`: '{}'".format(kwargs["sort"]))

            z0 = start

            for j in ids:
                pos[(i, j)] = (z0, z0 + z[j])
                z0 += z[j]

            start = end + pad


        def plot_arc(i, start, end, ocordI, plot_delayed=False):
            cls.ideogram_arc(start=start, end=end, radius=1.0, color=color,
                        width=width, alpha=alpha, ax=ax)

            start, end = pos[(i, i)]

            chord_color = chord_colors[i]

            # plot self-chords
            if mat[i, i] > 0:
                arc_color = chord_color
                if not arccolors is None:
                    arc_color = arccolors(i,i, mat[i,i])

                cls.self_chord_arc(start, end, radius=1 - width - gap,
                            chordwidth=0.7*chordwidth, color=arc_color,
                            alpha=alpha, ax=ax)

            # plot all other chords
            for j in range(ocordI):

                if not plot_delayed and j in delayedArcs:
                    continue


                cend = chord_colors[j]

                arc_color = chord_color
                if not arccolors is None:
                    arc_color = arccolors(i,j, mat[i,j])
                    cend = arccolors(j,i, mat[j,i])

                start1, end1 = pos[(i, j)]
                start2, end2 = pos[(j, i)]

                if mat[i, j] > 0 or mat[j, i] > 0:
                    cls.chord_arc(
                        start1, end1, start2, end2, radius=1 - width - gap,
                        chordwidth=chordwidth, color=arc_color, cend=cend,
                        alpha=alpha, ax=ax, use_gradient=use_gradient)


        # plot
        for i in range(len(x)):
            if not delayedArcs is None and i in delayedArcs:
                continue
            color = colors[i]
            # plot the arcs
            start, end = arc[i]
            plot_arc(i, start, end, i, False)

        if not delayedArcs is None:
            for i in range(len(x)):
                if not i in delayedArcs:
                    continue
                print("Delayed Arc", i)
                color = colors[i]
                # plot the arcs
                start, end = arc[i]
                plot_arc(i, start, end, len(x), True)

        # add names if necessary
        if names is not None:
            assert len(names) == num_nodes, "One name per node is required."

            prop = {
                "fontsize": kwargs.get("fontsize", 16*0.8),
                "ha": "center",
                "va": "center",
                "rotation_mode": "anchor"
            }

            for i, (pos, name, r) in enumerate(zip(nodePos, names, rotation)):
                rotate = rotate_names[i]
                pp = prop.copy()
                pp["color"] = fontcolor[i]

                if rotate:
                    angle  = np.average(arc[i])
                    rotate = 90

                    if 90 < angle < 180 or 270 < angle:
                        rotate = -90

                    if 90 < angle < 270:
                        pp["ha"] = "right"
                    else:
                        pp["ha"] = "left"
                elif r:
                    pp["va"] = "top"
                else:
                    pp["va"] = "bottom"

                name = "\n".join(wrap(name, 20))

                ax.text(pos[0], pos[1], name, rotation=pos[2] + rotate, **pp)

        # configure axis
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        ax.set_aspect(1)
        ax.axis('off')

        if show:
            plt.tight_layout()
            plt.show()

        return nodePos



    # ------------ #
    # Subfunctions #
    # ------------ #

    @classmethod
    def _get_normed_line(cls, mat, i, x, start, end, is_sparse):
        if is_sparse:
            row = mat.getrow(i).todense().A1
            return (row / x[i]) * (end - start)

        return (mat[i, :] / x[i]) * (end - start)


    @classmethod
    def polar2xy(cls, r, theta):
        '''
        Convert the coordinates of a point P from polar (r, theta) to cartesian
        (x, y).
        '''
        return np.array([r*np.cos(theta), r*np.sin(theta)])


    @classmethod
    def initial_path(cls, start, end, radius, width, factor=4/3):
        ''' First 16 vertices and 15 instructions are the same for everyone '''
        if start > end:
            start, end = end, start

        start *= np.pi/180.
        end   *= np.pi/180.

        # optimal distance to the control points
        # https://stackoverflow.com/questions/1734745/
        # how-to-create-circle-with-b#C3#A9zier-curves
        # use 16-vertex curves (4 quadratic Beziers which accounts for worst case
        # scenario of 360 degrees)
        inner = radius*(1-width)
        opt   = factor * np.tan((end-start)/ 16.) * radius
        inter1 = start*(3./4.)+end*(1./4.)
        inter2 = start*(2./4.)+end*(2./4.)
        inter3 = start*(1./4.)+end*(3./4.)

        verts = [
            cls.polar2xy(radius, start),
            cls.polar2xy(radius, start) + cls.polar2xy(opt, start+0.5*np.pi),
            cls.polar2xy(radius, inter1) + cls.polar2xy(opt, inter1-0.5*np.pi),
            cls.polar2xy(radius, inter1),
            cls.polar2xy(radius, inter1),
            cls.polar2xy(radius, inter1) + cls.polar2xy(opt, inter1+0.5*np.pi),
            cls.polar2xy(radius, inter2) + cls.polar2xy(opt, inter2-0.5*np.pi),
            cls.polar2xy(radius, inter2),
            cls.polar2xy(radius, inter2),
            cls.polar2xy(radius, inter2) + cls.polar2xy(opt, inter2+0.5*np.pi),
            cls.polar2xy(radius, inter3) + cls.polar2xy(opt, inter3-0.5*np.pi),
            cls.polar2xy(radius, inter3),
            cls.polar2xy(radius, inter3),
            cls.polar2xy(radius, inter3) + cls.polar2xy(opt, inter3+0.5*np.pi),
            cls.polar2xy(radius, end) + cls.polar2xy(opt, end-0.5*np.pi),
            cls.polar2xy(radius, end)
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

        return start, end, verts, codes

    @classmethod
    def ideogram_arc(cls, start, end, radius=1., width=0.2, color="r", alpha=0.7,
                    ax=None):
        '''
        Draw an arc symbolizing a region of the chord diagram.

        Parameters
        ----------
        start : float (degree in 0, 360)
            Starting degree.
        end : float (degree in 0, 360)
            Final degree.
        radius : float, optional (default: 1)
            External radius of the arc.
        width : float, optional (default: 0.2)
            Width of the arc.
        ax : matplotlib axis, optional (default: not plotted)
            Axis on which the arc should be plotted.
        color : valid matplotlib color, optional (default: "r")
            Color of the arc.

        Returns
        -------
        verts, codes : lists
            Vertices and path instructions to draw the shape.
        '''
        start, end, verts, codes = cls.initial_path(start, end, radius, width)

        opt    = 4./3. * np.tan((end-start)/ 16.) * radius
        inner  = radius*(1-width)
        inter1 = start*(3./4.) + end*(1./4.)
        inter2 = start*(2./4.) + end*(2./4.)
        inter3 = start*(1./4.) + end*(3./4.)

        verts += [
            cls.polar2xy(inner, end),
            cls.polar2xy(inner, end) + cls.polar2xy(opt*(1-width), end-0.5*np.pi),
            cls.polar2xy(inner, inter3) + cls.polar2xy(opt*(1-width), inter3+0.5*np.pi),
            cls.polar2xy(inner, inter3),
            cls.polar2xy(inner, inter3),
            cls.polar2xy(inner, inter3) + cls.polar2xy(opt*(1-width), inter3-0.5*np.pi),
            cls.polar2xy(inner, inter2) + cls.polar2xy(opt*(1-width), inter2+0.5*np.pi),
            cls.polar2xy(inner, inter2),
            cls.polar2xy(inner, inter2),
            cls.polar2xy(inner, inter2) + cls.polar2xy(opt*(1-width), inter2-0.5*np.pi),
            cls.polar2xy(inner, inter1) + cls.polar2xy(opt*(1-width), inter1+0.5*np.pi),
            cls.polar2xy(inner, inter1),
            cls.polar2xy(inner, inter1),
            cls.polar2xy(inner, inter1) + cls.polar2xy(opt*(1-width), inter1-0.5*np.pi),
            cls.polar2xy(inner, start) + cls.polar2xy(opt*(1-width), start+0.5*np.pi),
            cls.polar2xy(inner, start),
            cls.polar2xy(radius, start),
        ]

        codes += [
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]

        if ax is not None:
            path  = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=color, alpha=alpha,
                                    edgecolor=color, lw=cls.LW)
            ax.add_patch(patch)

        return verts, codes

    @classmethod
    def chord_arc(cls, start1, end1, start2, end2, radius=1.0, pad=2, chordwidth=0.7,
                ax=None, color="r", cend="r", alpha=0.7, use_gradient=False):
        '''
        Draw a chord between two regions (arcs) of the chord diagram.

        Parameters
        ----------
        start1 : float (degree in 0, 360)
            Starting degree.
        end1 : float (degree in 0, 360)
            Final degree.
        start2 : float (degree in 0, 360)
            Starting degree.
        end2 : float (degree in 0, 360)
            Final degree.
        radius : float, optional (default: 1)
            External radius of the arc.
        chordwidth : float, optional (default: 0.2)
            Width of the chord.
        ax : matplotlib axis, optional (default: not plotted)
            Axis on which the chord should be plotted.
        color : valid matplotlib color, optional (default: "r")
            Color of the chord or of its beginning if `use_gradient` is True.
        cend : valid matplotlib color, optional (default: "r")
            Color of the end of the chord if `use_gradient` is True.
        alpha : float, optional (default: 0.7)
            Opacity of the chord.
        use_gradient : bool, optional (default: False)
            Whether a gradient should be use so that chord extremities have the
            same color as the arc they belong to.

        Returns
        -------
        verts, codes : lists
            Vertices and path instructions to draw the shape.
        '''
        chordwidth2 = chordwidth

        dtheta1 = min((start1 - end2) % 360, (end2 - start1) % 360)
        dtheta2 = min((end1 - start2) % 360, (start2 - end1) % 360)

        start1, end1, verts, codes = cls.initial_path(start1, end1, radius, chordwidth)

        start2, end2, verts2, _ = cls.initial_path(start2, end2, radius, chordwidth)

        chordwidth2 *= np.clip(0.4 + (dtheta1 - 2*pad) / (15*pad), 0.2, 1)

        chordwidth *= np.clip(0.4 + (dtheta2 - 2*pad) / (15*pad), 0.2, 1)

        rchord  = radius * (1-chordwidth)
        rchord2 = radius * (1-chordwidth2)

        verts += [cls.polar2xy(rchord, end1), cls.polar2xy(rchord, start2)] + verts2

        verts += [
            cls.polar2xy(rchord2, end2),
            cls.polar2xy(rchord2, start1),
            cls.polar2xy(radius, start1),
        ]

        codes += [
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

        if ax is not None:
            path = Path(verts, codes)

            if use_gradient:
                # find the start and end points of the gradient
                points, min_angle = None, None

                if dtheta1 < dtheta2:
                    points = [
                        cls.polar2xy(radius, start1),
                        cls.polar2xy(radius, end2),
                    ]

                    min_angle = dtheta1
                else:
                    points = [
                        cls.polar2xy(radius, end1),
                        cls.polar2xy(radius, start2),
                    ]

                    min_angle = dtheta1

                # make the patch
                patch = patches.PathPatch(path, facecolor="none",
                                        edgecolor="none", lw=cls.LW)
                ax.add_patch(patch)  # this is required to clip the gradient

                # make the grid
                x = y = np.linspace(-1, 1, 100)
                meshgrid = np.meshgrid(x, y)

                cls.gradient(points[0], points[1], min_angle, color, cend, meshgrid,
                        patch, ax, alpha)
            else:
                patch = patches.PathPatch(path, facecolor=color, alpha=alpha,
                                        edgecolor=color, lw=cls.LW)

                idx = 16

                ax.add_patch(patch)

        return verts, codes

    @classmethod
    def linear_gradient(cls, cstart, cend, n=10):
        '''
        Return a gradient list of `n` colors going from `cstart` to `cend`.
        '''
        s = np.array(ColorConverter.to_rgb(cstart))
        f = np.array(ColorConverter.to_rgb(cend))

        rgb_list = [s + (t / (n - 1))*(f - s) for t in range(n)]

        return rgb_list

    @classmethod
    def gradient(cls, start, end, min_angle, color1, color2, meshgrid, mask, ax,
                alpha):
        '''
        Create a linear gradient from `start` to `end`, which is translationally
        invarient in the orthogonal direction.
        The gradient is then cliped by the mask.
        '''
        xs, ys = start
        xe, ye = end

        X, Y = meshgrid

        # get the distance to each point
        d2start = (X - xs)*(X - xs) + (Y - ys)*(Y - ys)
        d2end   = (X - xe)*(X - xe) + (Y - ye)*(Y - ye)

        dmax = (xs - xe)*(xs - xe) + (ys - ye)*(ys - ye)

        # blur
        smin = 0.015*len(X)
        smax = max(smin, 0.1*len(X)*min(min_angle/120, 1))

        sigma = np.clip(dmax*len(X), smin, smax)

        Z = gaussian_filter((d2end < d2start).astype(float), sigma=sigma)

        # generate the colormap
        n_bin = 100

        color_list = cls.linear_gradient(color1, color2, n_bin)

        cmap = LinearSegmentedColormap.from_list("gradient", color_list, N=n_bin)

        im = ax.imshow(Z, interpolation='bilinear', cmap=cmap,
                    origin='lower', extent=[-1, 1, -1, 1], alpha=alpha)

        im.set_clip_path(mask)

    @classmethod
    def self_chord_arc(cls, start, end, radius=1.0, chordwidth=0.7, ax=None,
                    color=(1,0,0), alpha=0.7):
        start, end, verts, codes = cls.initial_path(start, end, radius, chordwidth)

        rchord = radius * (1 - chordwidth)

        verts += [
            cls.polar2xy(rchord, end),
            cls.polar2xy(rchord, start),
            cls.polar2xy(radius, start),
        ]

        codes += [
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

        if ax is not None:
            path  = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=color, alpha=alpha,
                                    edgecolor=color, lw=cls.LW)
            ax.add_patch(patch)

        return verts, codes

    @classmethod
    def gamma_correct(cls, u):
        # Standard CRT Gamma
        GAMMA = 2.4
        if u > 0.00304:
            u = (1.055*u ** (1/GAMMA)) - 0.055
        else:
            u = 12.92*u

        return u


    @classmethod
    def hcl2rgb(cls, h,c,l):
        # ADAPTED FOR PYTHON BY MARKUS JOPPICH
        # 
        # HCL2RGB Convert a HCL (i.e., CIELUV) color space value to one
        #   in sRGB space.
        #   RGB = HCL2RGB(H, C, L) will convert the color (H, C, L) in
        #   HCL color space to RGB = [R, G, B] in sRGB color space.
        #   Values that lie outside sRGB space will be silently corrected.
        # Code written by Nicholas J. Hughes, 2014, released under the following
        # licence.
        #
        # The MIT License (MIT)
        #
        # Copyright (c) 2014 Nicholas J. Hughes
        # 
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        # 
        # The above copyright notice and this permission notice shall be included in
        # all copies or substantial portions of the Software.
        # 
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        # THE SOFTWARE.

        # D65 White Point
        WHITE_Y = 100.000
        WHITE_u = 0.1978398
        WHITE_v = 0.4683363

        if l < 0 or l > WHITE_Y or c < 0:
            print("Invalid CIE-HCL color.")
            assert(False)

        L = float(l)
        U = c * math.cos(math.radians(h))
        V = c * math.sin(math.radians(h))

        if L <= 0 and U == 0 and V == 0:
            X = 0
            Y = 0
            Z = 0
        else:
            Y = WHITE_Y
            if L > 7.999592:
                Y = Y*(((L + 16)/116) ** 3.0)
            else:
                Y = Y*L/903.3
            
            u = U/(13*L) + WHITE_u
            v = V/(13*L) + WHITE_v
            X = (9.0*Y*u)/(4*v)
            Z = -X/3 - 5*Y + 3*Y/v

        # Now convert to sRGB
        r = cls.gamma_correct((3.240479*X - 1.537150*Y - 0.498535*Z)/WHITE_Y)
        g = cls.gamma_correct((-0.969256*X + 1.875992*Y + 0.041556*Z)/WHITE_Y)
        b = cls.gamma_correct((0.055648*X - 0.204043*Y + 1.057311*Z)/WHITE_Y)

        # Round to integers and correct
        r = max([min([round(255 * r), 255]), 0])
        g = max([min([round(255 * g), 255]), 0])
        b = max([min([round(255 * b), 255]), 0])
        
        rgb = [x/255.0 for x in [r, g, b]]
        return rgb

    @classmethod
    def hue_pal(cls, n=1, h = [15, 375], c = 100, l = 65):
        assert(len(h) == 2)

        if ((h[1]-h[0] % 360) < 1):
            h[1] = h[1]-360/n

        hues = []
        curH = h[0]
        while curH < h[1]:
            hues.append((curH, c, l))
            curH += (h[1]-h[0])/n

            
        hexColors = []

        for x in hues:
            rgbColor = cls.hcl2rgb( x[0], x[1], x[2] )
            hexColor = to_hex(rgbColor)
            hexColors.append(hexColor)

        return hexColors


    @classmethod
    def getClusterColors(cls, clusterNames):

        numClusters = len(clusterNames)
        useColors = cls.hue_pal(numClusters)

        cluster2color = {}
        for x in range(0, numClusters):
            cluster2color[clusterNames[x].split(".")[1]] = useColors[x]

        return cluster2color


    @classmethod
    def sigmoid_curve(cls, p1, p2, resolution=0.1, smooth=0):
        x1, y1 = p1
        x2, y2 = p2
        
        xbound = 6 + smooth

        fxs = np.arange(-xbound,xbound+resolution, resolution)
        fys = expit(fxs)
        
        x_range = x2 - x1
        y_range = y2 - y1
        
        xs = x1 + x_range * ((fxs / (2*xbound)) + 0.5)
        ys = y1 + y_range * fys
        
        return xs, ys


    @classmethod 
    def sigmoid_arc(cls, p1, w1, p2, w2=None, resolution=0.1, smooth=0, ax=None):
        
        xs, ys1 = cls.sigmoid_curve(p1, p2, resolution, smooth)
        
        if w2 is None:
            w2 = w1
        
        p1b = p1[0], p1[1] - w1
        p2b = p2[0], p2[1] - w2

        xs, ys2 = cls.sigmoid_curve(p1b, p2b, resolution, smooth)
        
        return xs, ys1, ys2

    @classmethod
    def sankey(cls, flow_matrix=None, node_positions=None, link_alpha=0.5, colours=None, 
            colour_selection="source", resolution=0.1, smooth=0, **kwargs):
        #node_widths = [np.max([i, o]) for i, o in zip(in_totals, out_totals)]
        n = np.max(flow_matrix.shape)
        in_offsets = [0] * n
        out_offsets = [0] * n

        ax = kwargs.get("ax", plt.gca())
        
        for i, b1 in enumerate(node_positions):
            outputs = flow_matrix[i,:]
            for j, (w, b2) in enumerate(zip(outputs, node_positions)):
                if w:
                    p1 = b1[0], b1[1] - out_offsets[i]
                    p2 = b2[0], b2[1] - in_offsets[j]
                    xs, ys1, ys2 = cls.sigmoid_arc(p1, w, p2, resolution=resolution, smooth=smooth, ax=ax)
                    out_offsets[i] += w
                    in_offsets[j] += w
                
                    c = 'grey'

                    if type(colours) == str:
                        c = colours
                    elif type(colours) == list:
                        if colour_selection == "sink":
                            c = colours[j]
                        elif colour_selection == "source":
                            c = colours[i]
                    plt.fill_between(x=xs, y1=ys1, y2=ys2, alpha=link_alpha, color=c, axes=ax)

    @classmethod
    def generate_colormap(cls, N):
        arr = np.arange(N)/N
        N_up = int(np.ceil(N/7)*7)
        arr.resize(N_up)
        arr = arr.reshape(7,N_up//7).T.reshape(-1)
        ret = hsv(arr)
        n = ret[:,3].size
        a = n//2
        b = n-a
        for i in range(3):
            ret[0:n//2,i] *= np.arange(0.2,1,0.8/a)
        ret[n//2:,3] *= np.arange(1,0.1,-0.9/b)
    #     print(ret)
        return ret