import dabest
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np






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
    def plot_array_scatter(cls, fig, arr, discrete_legend=True):
        
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
            cnorm = matplotlib.colors.Normalize()
            cnorm.autoscale(arr)
        

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
            plt.colorbar(heatmap)





