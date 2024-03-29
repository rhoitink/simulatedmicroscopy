# Example data generation workflow

The generation of a simulated microscopy image consists of several steps:

1. Generation/loading of a point spread function (PSF)
2. Generation of a particle image (PI) from a set of coordinates
3. Convolution of the PI with the PSF
4. Optional: adding noise and downsampling the image

Each of the steps is explained below

## 1. Definition of a point spread function

There are two ways of getting a point spread function:

### 1.1 Reading in from Huygens `.h5` file format

Just pass the filename as an argument to the `HuygensPSF` class

```python
from simulatedmicroscopy import HuygensPSF

psf = HuygensPSF("filename.h5")
```

### 1.2 Generating a 3D Gaussian PSF

Pass the `sigma` values of the 3D Gaussian (in nanometers) to the the `GaussianPSF` together with the pixel_sizes (in meters). It's good to choose a smaller pixel size now and downsample your image later to reduce artifacts.

```python
from simulatedmicroscopy import GaussianPSF

# generate 3D Gaussian PSF with pixel sizes 50x10x10 nm^3 (z*y*x)
# and the sigma values for z,y,x resp. are 600, 250, 250 nm
psf = GaussianPSF([600., 250., 250.], pixel_sizes = [50e-9, 10e-9, 10e-9])
```

## 2. Particle image generation

Usually you'd read in coordinates from a file, any (N,3) shaped `list`/`numpy.ndarray` will work here. Coordinates are given in `x,y,z` order.

```python
from simulatedmicroscopy import Image, Coordinates, Sphere

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

# create a spherical particle (radius = 100 nm) that will be placed at the coordinates
# other options are listed in `simulatedmicroscopy.particle`
particle = Sphere(ps.get_pixel_sizes(), radius = 100e-9)

# create image by placing this particle at set coordinates
image = Image.create_particle_image(coords, particle)

```

You now have a particle image in the `image` variable.

The coordinates of the particles are stored as pixel indices (in `zyx` order), to retrieve them, run:

```python
pixel_indices = image.get_pixel_coordinates()

print(pixel_indices[0]) # position of first particle
```

## 3. Convolution of your image with the PSF

When you have both the PI and PSF, you can convolve the first with the latter, by running:

```python
image.convolve(psf)
```

You convolved image is now saved in the `image` variable.

## 4. Optional: adding noise and downsampling the image

Adding noise and downsampling the image to a lower pixel size is included as well and can be done as follows:

```python
# downsample by a factor 2 in z and factor 3 in xy
# add Poisson noise (shot noise) with a signal-to-noise ratio of 30
# add Gaussian noise (read noise, additive) with a mean value of 1e-5 and a signal-to-noise ratio of 50
image.downsample([2, 3, 3]).add_shot_noise(SNR = 30.0).add_read_noise(SNR = 50.0, background = 1e-5)
```

The downsampling by a factor two along a dimension, means that the number of pixels along that dimension is divided by two, which leads to an increase of the pixel size by a factor or two.

## Saving your image

You can save your image at any step along the way by running

```python
image.save_h5file("filename.h5")
```

## Loading the image

The image (and particle coordinates) can then later be retrieved again by:

```python
from simulatedmicroscopy import Image

image = Image.load_h5file("filename.h5")
pixel_indices = image.get_pixel_coordinates()
```
