# Example usage

The generation of a simulated microscopy image consists of several steps:

1. Generation/loading of a point spread function (PSF)
2. Generation of a point source image (PSI) from a set of coordinates
3. Convolution of the PSI with the PSF
4. Optional: adding noise and downsampling the image

Each of the steps is explained below

## 1. Definition of a point spread function

There are two ways of getting two a point spread function:

### 1. Reading in from Huygens `.h5` file format

Just pass the filename as an argument to the `HuygensPSF` class

```python
from simulatedmicroscopy import HuygensPSF

psf = HuygensPSF("filename.h5")
```

### 2. Generating a 3D Gaussian PSF

Pass the `sigma` values of the 3D Gaussian (in nanometers) to the the `GaussianPSF` together with the pixel_sizes (in meters). It's good to choose a smaller pixel size now and downsample your image later to reduce artifacts.

```python
from simulatedmicroscopy import GaussianPSF

# generate 3D Gaussian PSF with pixel sizes 50x10x10 nm^3 (z*y*x)
# and the sigma values for z,y,x resp. are 600, 250, 250 nm
psf = GaussianPSF([600., 250., 250.], pixel_sizes = [50e-9, 10e-9, 10e-9])
```

## 2. Point source image generation

Usually you'd read in coordinates from a file, any (N,3) shaped `list`/`numpy.ndarray` will work here. Coordinates are given in `x,y,z` order.

```python
from simulatedmicroscopy import Image, Coordinates

# read in coordinates from a file
# for now, just some dummy coordinates
coords = Coordinates(
    [
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
        [2.,0.,0.],
        [0.,2.,0.],
        [0.,0.,2.],
    ]
)

# create image with same pixel size as the psf
image = Image.create_point_image(coords, 
                            pixel_sizes = psf.get_pixel_sizes())

# it's good to realize that this image adds some spacing (0.5 µm) on all sides
```

You now have a point source image in the `image` variable.

The coordinates of the particles are stored as pixel indices (in `zyx` order), to retrieve them, run:

```python
pixel_indices = image.get_pixel_coordinates()

print(pixel_indices[0]) # position of first particle
```

## 3. Convolution of your image with the PSF

When you have both the PSI and PSF, you can convolve the first with the latter, by running:

```python
image.convolve(psf)
```

You convolved image is now saved in the `image` variable.

## 4. Optional: adding noise and downsampling the image

Adding Poisson noise and downsampling the image to a lower pixel size is included as well and can be done as follows:

```python
# add Poisson noise
# downsample by a factor 2 in z and factor 3 in xy
image.noisify().downsample([2, 3, 3])
```

The downsampling by a factor two along a dimension, means that the number of pixels along that dimension is divided by two, which leads to an increase of the pixel size by a factor or two.

## Saving your image

You can save your image at any step along the way by running

```python
image.save_h5file("filename.h5")
```

## Loading the image

The image can then later be retrieved again by:

```python
from simulatedmicroscopy import Image

image = Image.load_h5file("filename.h5")
```
