import numpy as np
from PIL import Image

class PaletteConverter:
    """
    A class to convert images to index masks using a given palette.

    This class allows converting images to index masks by mapping unique colors
    in the input image to corresponding object indices in the output index mask.
    The palette provided during initialization is used to assign colors to the objects.
    Color black is assumed to be background and is ignored, thus index_mask's indices start with 1.

    Recommended to use over a set of images (e.g. masks for a single video), 
    as it provides CONSISTENT object indices even if some of them are not present in certain frames.

    Parameters:
    -----------
    palette : bytes
        A bytes object representing the palette used for color mapping. See `util.palette` module.
    num_potential_colors : int, optional
        The number of potential colors in the lookup table. Default is 256.

    Properties:
    -----------
    palette : bytes
        The palette used for color mapping.
    lookup : numpy.ndarray
        An array to keep track of color-to-object index mapping.
    num_objects : int
        The number of unique objects detected in all the images.

    Methods:
    --------
    image_to_index_mask(img: Image.Image) -> Image.Image:
        Convert an input image to an index mask using the palette and stored object mapping.

    Example:
    --------
    # Create a palette converter object
    ```
    palette = b'\\xff\\x00\\x00\\xff\\xff\\xff\\x00\\x00\\x00\\x00\\xff\\x00' # or use palettes from `util.palette` module
    converter = PaletteConverter(palette)

    # Convert an image to index mask
    from PIL import Image
    for img_path in [...]:
        input_img = Image.open(img_path)
        index_mask = converter.image_to_index_mask(input_img)  # lookup gets updated internally to preserve consistent object indices across multiple images
    ```
    """
    def __init__(self, palette: bytes, num_potential_colors=256) -> None:
        self._palette = palette
        self._lookup = np.zeros(num_potential_colors, dtype=np.uint8)
        self._num_objects = 0

    def image_to_index_mask(self, img: Image.Image) -> Image.Image:
        img_p = img.convert('P')
        unique_colors = img_p.getcolors()
        for _, c in unique_colors:
            if c == 0:
                # Blacks is always 0 and is ignored 
                continue
            elif self._lookup[c] == 0:
                self._num_objects += 1
                self._lookup[c] = self._num_objects

        # We need range indices like (0, 1, 2, 3, 4) for each unique color, in any order, as long as black is still 0
        # If the new colors appear, we'll treat them as new objects
        index_array = self._lookup[img_p]  # We use the lookup as the "image", and the actual P images as the "indices" for it (thus color_id -> object_id)
        index_mask = Image.fromarray(index_array, mode='P')
        index_mask.putpalette(self._palette)

        return index_mask

    @property
    def palette(self):
        return self._palette
    
    @property
    def lookup(self):
        return self._lookup
    
    @property
    def num_objects(self):
        return self._num_objects
