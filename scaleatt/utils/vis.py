import numpy as np
import matplotlib.pyplot as plt
import typing

def plot_images_in_actual_size_color(imgs: typing.List[np.ndarray], titles: typing.List[str], rows: int) -> None:
    """
    Assumes that all images in list have same size. Otherwise, images are scaled.
    :param imgs: list of images
    :param titles: list of titles for respective images
    :param rows: the number of rows for the sub figs
    """

    margin = 50  # pixels
    spacing = 35  # pixels
    dpi = 100.  # dots per inch

    cols: int = int(np.ceil(len(imgs) / rows))

    width = (imgs[0].shape[1] * cols + 2 * margin + spacing) / dpi  # inches
    height = (imgs[0].shape[0] * rows + 2 * margin + spacing) / dpi

    left = margin / dpi / width  # axes ratio
    bottom = margin / dpi / height
    wspace = spacing / float(200)

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1. - left, top=1. - bottom,
                        wspace=wspace, hspace=wspace)

    for ax, im, name in zip(axes.flatten(), imgs, titles):
        ax.axis('off')
        ax.set_title('{}'.format(name))
        ax.imshow(im)

    plt.show()