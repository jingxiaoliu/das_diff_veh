import copy

import random
from apis.dispersion_classes import SurfaceWaveDispersion
from apis.virtual_shot_gather import VirtualShotGather
from modules.utils import fv_map_enhance,plot_fv_map

def save_disp_imgs(windows, weight, min_win, x, start_x, end_x, offset, fig_dir):
    # Create an instance of VirtualShotGathersFromWindows class
    image_from_window_cls = VirtualShotGathersFromWindows    
    # Randomly select 'min_win' elements from 'windows'
    sel_idx = random.sample(range(len(windows)), min_win)    
    # Create an instance of VirtualShotGathersFromWindows using all 'windows'
    images_all = image_from_window_cls(windows)    
    # Create an instance of VirtualShotGathersFromWindows using selected 'windows'
    _images = image_from_window_cls([e for i, e in enumerate(windows) if i in sel_idx])    
    # Retrieve images from '_images' instance
    _images.get_images(pivot=x, start_x=start_x, end_x=end_x, wlen=2, include_other_side=True)    
    # Plot average image
    _images.avg_image.plot_image(norm=True, x_lim=[-offset, offset], 
                                 fig_dir=f"{fig_dir}/{x}/", fig_name=f"sg_{weight}_cars.png")    
    # Compute dispersion image
    _images.avg_image.compute_disp_image(end_x=0, start_x=-offset)    
    # Enhance the fv_map of dispersion
    fv_map_enhanced = fv_map_enhance(_images.avg_image.disp.fv_map)
    
    # Plot fv_map without normalization
    plot_fv_map(_images.avg_image.disp.fv_map, _images.avg_image.disp.freqs, 
                _images.avg_image.disp.vels, norm=False,
                fig_dir=f"{fig_dir}/{x}/", fig_name=f"disp_{weight}_cars_no_norm.png",
                ridge_data=None, norm_part=False)    
    # Plot fv_map with normalization
    plot_fv_map(_images.avg_image.disp.fv_map, _images.avg_image.disp.freqs, 
                _images.avg_image.disp.vels, norm=True,
                fig_dir=f"{fig_dir}/{x}/", fig_name=f"disp_{weight}_cars_no_enhance.png",
                ridge_data=None, norm_part=False)    
    # Plot enhanced fv_map
    plot_fv_map(fv_map_enhanced, _images.avg_image.disp.freqs, 
                _images.avg_image.disp.vels, norm=True,
                fig_dir=f"{fig_dir}/{x}/", fig_name=f"disp_{weight}_cars.png",
                ridge_data=None, norm_part=False)
    
    return images_all

class ImagesFromWindows:
    def __init__(self, windows, image_cls):
        """

        :param windows: List of SurfaceWaveWindow obj
        """
        self.windows = windows
        self.image_cls = image_cls

    def get_images(self, norm=False, mute_offset=300, mute=True, **imaging_kwargs):
        self.images = []

        for k, window in enumerate(self.windows):
            if mute and not window.muted_along_traj:
                window = copy.deepcopy(window)
                window.mute_along_traj(offset=mute_offset)
            image = self.image_cls(window, norm=norm, **imaging_kwargs)
            self.images.append(image)

        self.avg_image = sum(self.images)
        self.avg_image = self.avg_image / len(self.images)


    def save_images(self, fig_folder, file_prefix):

        for k, image in enumerate(self.images):
            fname = f"{file_prefix}{k}.png"
            image.plot_image(fname, norm=True, fig_folder=fig_folder)

        fname = f"{file_prefix}_avg.png"
        self.avg_image.plot_image(fname, norm=True, fig_folder=fig_folder)


class DispersionImagesFromWindows(ImagesFromWindows):

    def __init__(self, windows, image_cls=SurfaceWaveDispersion):
        super().__init__(windows, image_cls)

    def save_images(self, fig_folder, file_prefix='veh_disp'):
        super(DispersionImagesFromWindows, self).save_images(fig_folder, file_prefix)


class VirtualShotGathersFromWindows(ImagesFromWindows):
    def __init__(self, windows, image_cls=VirtualShotGather):
        """

        :param windows: List of SurfaceWaveWindow obj
        """
        super().__init__(windows, image_cls)

    def get_images(self, norm=False, mute_offset=300, mute=False, **imaging_kwargs):
        super().get_images(norm=False, mute_offset=300, mute=False, **imaging_kwargs)

    def save_images(self, fig_folder, file_prefix='veh_vshot'):
        super(VirtualShotGathersFromWindows, self).save_images(fig_folder, file_prefix)
