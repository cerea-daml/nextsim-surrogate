
def interpolate(dataset, data_name, target_shape, d, inds):
        
    tmp = {}

    tmp[data_name] = []
    var = dataset[data_name].values.flatten()[inds]
    var.shape = target_shape.shape
        #var = block_reduce(
         #       var[91:, 8:-8], block_size=(self.N_res, self.N_res), func=np.mean
          #  )
    tmp[data_name].append(var)

    return tmp

def lon_lat_to_cartesian(lon, lat):
    # WGS 84 reference coordinate system parameters
    A = 6378.137  # major axis [km]
    E2 = 6.69437999014e-3  # eccentricity squared

    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
        # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)
    return x, y, z

class _HeatMapper2(object):
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cellsize, cellsize_vmax,
                 cbar, cbar_kws,
                 xticklabels=True, yticklabels=True, mask=None, ax_kws=None, rect_kws=None):
        """Initialize the plotting object."""
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)

        # Validate the mask and convet to DataFrame


        # Get good names for the rows and columns
        xtickevery = 1
        x_axis_labels = ['1','2','3','4','5','6','7','8','9','10','11','12'] # labels for x-axis
        y_axis_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] # labels for y-axis

        # Get the positions and used label for the ticks
        nx, ny = data.T.shape
        
        self.xticks = 0.5 + np.arange(12)
        self.xticklabels = x_axis_labels
        
        self.yticks = 0.5 + np.arange(12)
        self.yticklabels = y_axis_labels
        
        # Get good names for the axis labels
        self.ylabel = 'Initialization time'
        self.xlabel = 'Lead time (months)'
                
        # Determine good default values for cell size
        self._determine_cellsize_params(plot_data, cellsize, cellsize_vmax)
        
         
        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data

        # Sort out the annotations
        if annot is None:
            annot = False
            annot_data = None
        elif isinstance(annot, bool):
            if annot:
                annot_data = plot_data
            else:
                annot_data = None
        else:
            try:
                annot_data = annot.values
            except AttributeError:
                annot_data = annot
            if annot.shape != plot_data.shape:
                raise ValueError('Data supplied to "annot" must be the same '
                                 'shape as the data to plot.')
            annot = True
        self.annot = annot
        self.annot_data = annot_data
       
        self.cmap = cmap
        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws
        self.annot_kws.setdefault('color', "black")
        self.annot_kws.setdefault('ha', "center")
        self.annot_kws.setdefault('va', "center")
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws
        self.cbar_kws.setdefault('ticks', mpl.ticker.MaxNLocator(6))
        self.ax_kws = {} if ax_kws is None else ax_kws
        self.rect_kws = {} if rect_kws is None else rect_kws
        #self.rect_kws.setdefault('edgecolor', "black")
        self.vmax = vmax
        self.vmin = vmin
        self.title = title
    
    def _determine_cellsize_params(self, plot_data, cellsize, cellsize_vmax):
        if isinstance(cellsize, pd.DataFrame):
            cellsize = cellsize.values
        self.cellsize = cellsize
        if cellsize_vmax is None:
            cellsize_vmax = cellsize.max()
        self.cellsize_vmax = cellsize_vmax
        
        
    def plot(self, data, ax, cax):
        """Draw the heatmap on the provided Axes."""

        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # Draw the heatmap and annotate
        height, width = self.plot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)

        #data = self.plot_data.data
        array = self.data
        cellsize = self.cellsize
        # Draw rectangles instead of using pcolormesh
        # Might be slower than original heatmap
        annot_data = self.annot_data
        if not self.annot:
            annot_data = np.zeros(self.plot_data.shape)


        for x, y, val, s, an_val in zip(xpos.flat, ypos.flat, data.flat, cellsize.flat, annot_data.flat):
            
            
            vv = (val - self.vmin) / (self.vmax - self.vmin)
            size = np.clip(s / self.cellsize_vmax, 0.1, 1.0)
            color = self.cmap(vv)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, **self.rect_kws)
            ax.add_patch(rect)

            if self.annot:
                annotation = ("{}").format(str(an_val)[:4])
                text = ax.text(x, y, annotation, **self.annot_kws)
                    # add edge to text
                text_luminance = relative_luminance(text.get_color())
                text_edge_color = ".15" if text_luminance > .408 else "w"
                text.set_path_effects([mpl.patheffects.withStroke(linewidth=1, foreground=text_edge_color)])

          
        # Set the axis limits
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))

        # Set other attributes
        ax.set(**self.ax_kws)

        if self.cbar:
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            scalar_mappable = mpl.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            scalar_mappable.set_array(self.plot_data.data)
            cb = ax.figure.colorbar(scalar_mappable, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            font_size = 18 # Adjust as appropriate.
            cb.ax.tick_params(labelsize=font_size)
            # if kws.get('rasterized', False):
            #     cb.solids.set_rasterized(True)

        # Add row and column labels
        self.xticks = np.arange(12) + 0.5
        self.xticklabels = x_axis_labels
        
        self.yticks = np.arange(12) + 0.5
        self.yticklabels = y_axis_labels
        
        ax.set(xticks=self.xticks, yticks=self.yticks)
        xtl = ax.set_xticklabels(self.xticklabels,fontsize = 18)
        ytl = ax.set_yticklabels(self.yticklabels, rotation="vertical",fontsize = 18)

        # Possibly rotate them if they overlap
        ax.figure.draw(ax.figure.canvas.get_renderer())
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Add the axis labels
        ax.set_title(self.title, fontsize = 20)
        ax.set_xlabel(self.xlabel, fontsize = 20)
        ax.set_ylabel(self.ylabel, fontsize = 20)
        # Invert the y axis to show the plot in matrix form
        ax.invert_yaxis()

def heatmap2(array, title, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g", annot_kws=None,
            cellsize=None, cellsize_vmax=None,
            cbar=True, cbar_kws=None, cbar_ax=None,
            square=False, xticklabels="auto", yticklabels="auto",
            mask=None, ax=None, ax_kws=None, rect_kws=None):

    # Initialize the plotter object
    plotter = _HeatMapper2(array, vmin, vmax, cmap, center, robust,
                          annot, fmt, annot_kws,
                          cellsize, cellsize_vmax,
                          cbar, cbar_kws, xticklabels,
                          yticklabels, mask, ax_kws, rect_kws)

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")

    # delete grid
    ax.grid(alpha = 0.3, ls = "--", lw = 1)
    plotter.plot(array, ax, cbar_ax)
    return ax