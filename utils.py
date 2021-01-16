import re
import tifffile
import xmltodict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import holoviews as hv
import param


def get_metadata_as_series(filepath, regex, filename_key="filepath"):
    '''
    provided with a filepath (can be a string or a pathlib.PosixPath object),
    tries to match the path against the regular expression regex.
    The extracted keys, plus the filepath are returned as a pandas Series object
    '''
    filepath = str(filepath)
    m = re.match(regex, filepath)
    if m is not None:
        tmp = m.groupdict()
        tmp[filename_key] = filepath
        if 'MIBI' in filepath:
            tmp['Channels'] = [x['channel.target'] for x in tifffile.TiffFile(filepath).shaped_metadata]
#             tmp['Channels'] = [x.description.split('"channel.target": "')[1].split('", "shape"')[0] for x in tifffile.TiffFile(filepath).pages]
        else:
            tmp['Channels'] = [xmltodict.parse(x.tags['ImageDescription'].value)['PerkinElmer-QPI-ImageDescription']['Name'] for x in tifffile.TiffFile(filepath).pages[:-1]]
#             tmp['Channels'] = [x.description.split('<Name>')[1].split(' ')[0].split('<')[0].split('+')[0] for x in  tifffile.TiffFile(filepath).pages[:-1]]
        tmp['imageID'] = "_".join((tmp['Modality'], str(tmp["Pt"]),str(tmp["Point"])))
        tmp['pairID'] = '_'.join((str(tmp['Pt']), str(tmp['Point'])))
        return pd.Series(tmp)
    else:
        print(f"Extracting metadata for {filepath} failed.")
        return None



def mult_ch_pairs_plt_grd(img_pairs, common_ch_names):
    '''
    Function that takes in a pair of multiband images, and a dictionary of channel names, and
    constructs a grid of plots with each image channel pair and histogram of channel
    values.
    '''
    # Number of channels to set number of rows in grid
    n_ch = img_pairs[0].shape[0]

    # Lists of channel names with modality identification
    ch = common_ch_names

    fig, ax = plt.subplots(n_ch, 4, figsize=(30, 45))

    for i in range(n_ch):
        # histogram 1
        sns.distplot(img_pairs[0][i].ravel(), kde=False, ax=ax[i, 0])
        ax[i, 0].axvline(np.mean(img_pairs[0][i].ravel()), color='r')
        ax[i, 0].set_yscale('log', nonposy='clip')
        mn = np.mean(img_pairs[0][i].ravel())
        ax[i, 0].set_title('Log yscale: Mean = ' + "%.2f" % mn)

        # image 1
        imgFig = ax[i, 1].imshow(img_pairs[0][i], interpolation='none')
        fig.colorbar(imgFig, ax=ax[i, 1])
        ax[i, 1].set_title(ch[i])
        ax[i, 1].axis('off')

        # image 2
        imgFig = ax[i, 2].imshow(img_pairs[1][i], interpolation='none')
        fig.colorbar(imgFig, ax=ax[i, 2])
        ax[i, 2].set_title(ch[i])
        ax[i, 2].axis('off')

        # histogram 2
        sns.distplot(img_pairs[1][i].ravel(), kde=False, ax=ax[i, 3])
        ax[i, 3].axvline(np.mean(img_pairs[1][i].ravel()), color='r')
        ax[i, 3].set_yscale('log', nonposy='clip')
        mn = np.mean(img_pairs[1][i].ravel())
        ax[i, 3].set_title('Log yscale: Mean = ' + "%.2f" % mn)

    return (plt.show())

# Set hover background color to marker colors in plotly scatter3d
# source https://community.plotly.com/t/hover-background-color-on-scatter-3d/9185/9
def hex_to_rgb(value):
    """Convert a hex-formatted color to rgb, ignoring alpha values."""
    value = value.lstrip("#")
    return [int(value[i:i + 2], 16) for i in range(0, 6, 2)]


def rbg_to_hex(c):
    """Convert an rgb-formatted color to hex, ignoring alpha values."""
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def get_colors_for_vals(vals, vmin, vmax, colorscale, return_hex=True):
    """Given a float array vals, interpolate based on a colorscale to obtain
    rgb or hex colors. Inspired by
    `user empet's answer in \
    <community.plotly.com/t/hover-background-color-on-scatter-3d/9185/6>`_."""
    from numbers import Number
    from ast import literal_eval

    if vmin >= vmax:
        raise ValueError("`vmin` should be < `vmax`.")

    if (len(colorscale[0]) == 2) and isinstance(colorscale[0][0], Number):
        scale, colors = zip(*colorscale)
    else:
        scale = np.linspace(0, 1, num=len(colorscale))
        colors = colorscale
    scale = np.asarray(scale)

    if colors[0][:3] == "rgb":
        colors = np.asarray([literal_eval(color[3:]) for color in colors],
                            dtype=np.float_)
    elif colors[0][0] == "#":
        colors = np.asarray(list(map(hex_to_rgb, colors)), dtype=np.float_)
    else:
        raise ValueError("This colorscale is not supported.")

    colorscale = np.hstack([scale.reshape(-1, 1), colors])
    colorscale = np.vstack([colorscale, colorscale[0, :]])
    colorscale_diffs = np.diff(colorscale, axis=0)
    colorscale_diff_ratios = colorscale_diffs[:, 1:] / colorscale_diffs[:, [0]]
    colorscale_diff_ratios[-1, :] = np.zeros(3)

    vals_scaled = (vals - vmin) / (vmax - vmin)

    left_bin_indices = np.digitize(vals_scaled, scale) - 1
    left_endpts = colorscale[left_bin_indices]
    vals_scaled -= left_endpts[:, 0]
    diff_ratios = colorscale_diff_ratios[left_bin_indices]

    vals_rgb = (
            left_endpts[:, 1:] + diff_ratios * vals_scaled[:, np.newaxis] + 0.5
    ).astype(np.uint8)

    if return_hex:
        return list(map(rbg_to_hex, vals_rgb))
    return [f"rgb{tuple(v)}" for v in vals_rgb]




# # load image based on selections
# def panel_images(pairID, ch_choice, metaDF):
#     imgPair = list(metaDF[metaDF['pairID'] == pairID]['filepath'])
#     mibiImg = tifffile.imread(imgPair[0], key=com_ch_lut[ch_choice]['MIBI'])
#     polarisImg = tifffile.imread(imgPair[1],
#                                  key=com_ch_lut[ch_choice]['Polaris'])
#
#     mibi = hv.Image(mibiImg).options(tools=['hover'], cmap="gray", width=400,
#                                      height=400, colorbar=False)
#     polaris = hv.Image(polarisImg).options(tools=['hover'], cmap="gray",
#                                            width=400, height=400,
#                                            colorbar=False)
#
#     mHist_F, mHist_E = np.histogram(mibiImg.ravel(), 50)
#     pHist_F, pHist_E = np.histogram(polarisImg.ravel(), 50)
#
#     mibiDist = hv.Histogram((mHist_E, mHist_F)).options(width=400, height=400)
#     polarisDist = hv.Histogram((pHist_E, pHist_F)).options(width=400,
#                                                            height=400)
#
#     layout = hv.Layout(mibiDist + mibi + polaris + polarisDist).cols(4)
#
#     layout.opts(shared_axes=False)
#     return layout


# if __name__ == '__main__':
#     app.run()
