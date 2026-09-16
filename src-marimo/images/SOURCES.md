# Image sources and licensing

This folder holds shared raster assets for the `src-marimo` notebooks. Two kinds of
image live here, with different provenance:

- The four **standard test images** at the top level (`astronaut.png`, `coffee.png`,
  `chelsea.png`, `raccoon.png`) — reusable across notebooks (color spaces, filtering,
  histogram equalization, ...).
- The **textbook figure extracts** in `textbook_figures/` — used by
  `07_sensors_and_color.py`, `11_multiresolution_representations.py`, and
  `12_geometric_transformations.py`, cropped from the course PDF for citation purposes.

## Standard test images

The first three were fetched from [scikit-image](https://scikit-image.org/)'s bundled
sample data (`skimage.data`, version 0.26.0), saved as lossless PNG. scikit-image
documents the source and license of each image in its own docstrings, reproduced here:

### `astronaut.png`
- **Function:** `skimage.data.astronaut()`
- **Subject:** Eileen Collins, American astronaut (first woman to pilot and command a
  Space Shuttle).
- **Source:** Downloaded from the NASA Great Images database (<https://flic.kr/p/r9qvLn>).
- **License:** No known copyright restrictions — released into the public domain.
- **Dimensions:** 512 × 512, RGB.

### `coffee.png`
- **Function:** `skimage.data.coffee()`
- **Subject:** A cup of coffee on a wooden table (elliptical shapes, mixed smooth/coarse
  texture).
- **Source:** Photograph courtesy of Pikolo Espresso Bar.
- **License:** CC0 (no copyright restrictions), photographer Rachel Michetti.
- **Dimensions:** 400 × 600, RGB.

### `chelsea.png`
- **Function:** `skimage.data.chelsea()` (alias `skimage.data.cat()`)
- **Subject:** "Chelsea the cat" — texture, prominent edges at multiple scales and
  orientations.
- **License:** CC0 (no copyright restrictions), photographer Stefan van der Walt.
- **Dimensions:** 300 × 451, RGB.

### `raccoon.png`
- **Function:** `scipy.datasets.face()` (the function is named after "face" but the
  subject is a raccoon; fetched via `scipy`, not `skimage.data`, and requires the
  optional `pooch` dependency to download).
- **Subject:** A raccoon peeking out of green palm-frond foliage — added specifically
  for its strong green content (green channel has the highest mean of the three
  channels), which the first three images lack.
- **Source:** Per the function's own docstring, "derived from
  <https://pixnio.com/fauna-animals/raccoons/raccoon-procyon-lotor>" — Pixnio is a
  public-domain stock photo site (images released with no known copyright
  restrictions).
- **License:** Public domain, per the source above.
- **Dimensions:** 768 × 1024, RGB. (Note: larger than the other three — left at its
  native resolution rather than resized, consistent with not modifying the others.)
- This is also a long-standing "standard test image" in its own right — it's shipped
  with SciPy/NumPy for exactly this purpose and shows up throughout their
  documentation and tutorials.

Why these and not the classic "Lena" test image: Lena is a 1972 Playboy centerfold crop
whose continued use in course materials and papers has drawn increasing (and
well-founded) criticism; the images above give the same "standard test image" role with
clean, unambiguous public-domain/CC0 licensing.

## Textbook figure extracts (`textbook_figures/`)

Cropped directly from `Szeliski_CVAABook_2ndEd.pdf` (Richard Szeliski, *Computer Vision:
Algorithms and Applications*, 2nd ed., final draft Sept. 2021, Springer) using PyMuPDF,
for citation in `07_sensors_and_color.py`, `11_multiresolution_representations.py`, and
`12_geometric_transformations.py`. Reproduced here for educational, non-commercial
classroom use with full attribution, consistent with the book's own citation of its
sources.

### `szeliski_fig2_23_sensing_pipeline.png`
- **Book figure:** Figure 2.23, p. 80 (PDF page index 105).
- **Caption:** "Image sensing pipeline, showing the various sources of noise as well as
  typical digital post-processing steps."
- **Original source:** Szeliski's own diagram (a vector drawing in the PDF, not itself
  attributed to a third party in the caption).

### `szeliski_fig2_24a_ccd_cmos.png`
- **Book figure:** Figure 2.24(a), p. 81 (PDF page index 106).
- **Caption:** "CCDs move photogenerated charge from pixel to pixel and convert it to
  voltage at the output node; CMOS imagers convert charge to voltage inside each pixel."
- **Original source, per the book's own citation:** Litwiller (2005), © 2005 Photonics
  Spectra. Reproduced in the textbook and re-extracted here for the same educational
  purpose; the original copyright holder is Photonics Spectra.

### `szeliski_fig2_24b_cmos_cutaway.png`
- **Book figure:** Figure 2.24(b), p. 81 (PDF page index 106).
- **Caption:** "cutaway diagram of a CMOS pixel sensor."
- **Original source, per the book's own citation:**
  <https://micro.magnet.fsu.edu/primer/digitalimaging/cmosimagesensors.html> (Molecular
  Expressions / Florida State University).

### `szeliski_fig3_29_signal_decimation.png`
- **Book figure:** Figure 3.29, p. 153 (PDF page index 178).
- **Caption:** "Signal decimation: (a) the original samples are (b) convolved with a
  low-pass filter before being downsampled."
- **Original source:** Szeliski's own diagram (a vector drawing in the PDF, not itself
  attributed to a third party in the caption).

### `szeliski_fig3_30_decimation_filter_responses.png`
- **Book figure:** Figure 3.30, p. 155 (PDF page index 180).
- **Caption:** "Frequency response for some 2× decimation filters. The cubic a = −1
  filter has the sharpest fall-off but also a bit of ringing; the wavelet analysis
  filters (QMF-9 and JPEG 2000), while useful for compression, have more aliasing."
- **Original source:** Szeliski's own plot (not itself attributed to a third party in
  the caption).

### `szeliski_fig3_31_traditional_image_pyramid.png`
- **Book figure:** Figure 3.31, p. 156 (PDF page index 181).
- **Caption:** "A traditional image pyramid: each level has half the resolution (width
  and height), and hence a quarter of the pixels, of its parent level."
- **Original source:** Szeliski's own diagram (not itself attributed to a third party in
  the caption).

### `szeliski_fig3_32_gaussian_pyramid_signal_diagram.png`
- **Book figure:** Figure 3.32, p. 156 (PDF page index 181).
- **Caption:** "The Gaussian pyramid shown as a signal processing diagram: The (a)
  analysis and (b) re-synthesis stages are shown as using similar computations. The
  white circles indicate zero values inserted by the ↑2 upsampling operation. Notice
  how the reconstruction filter coefficients are twice the analysis coefficients."
- **Original source:** Szeliski's own diagram (not itself attributed to a third party in
  the caption).

### `szeliski_fig3_34_dog_bandpass.png`
- **Book figure:** Figure 3.34, p. 158 (PDF page index 183).
- **Caption:** "The difference of two low-pass filters results in a band-pass filter.
  The dashed blue lines show the close fit to a half-octave Laplacian of Gaussian."
- **Original source:** Szeliski's own diagram (not itself attributed to a third party in
  the caption).

### `szeliski_fig3_44_transformation_hierarchy.png`
- **Book figure:** Figure 3.44, p. 169 (PDF page index 194).
- **Caption:** "Basic set of 2D geometric image transformations."
- **Original source:** Szeliski's own diagram (not itself attributed to a third party in
  the caption).

### `szeliski_fig3_45_forward_warping.png`
- **Book figures:** Figure 3.45 and Algorithm 3.1, p. 170 (PDF page index 195).
- **Caption:** "Forward warping algorithm: (a) a pixel f(x) is copied to its
  corresponding location x' = h(x) in image g(x'); (b) detail of the source and
  destination pixel locations." Algorithm 3.1 (the accompanying pseudocode) is included
  in the same crop.
- **Original source:** Szeliski's own diagram (not itself attributed to a third party in
  the caption).

### `szeliski_fig3_46_inverse_warping.png`
- **Book figures:** Figure 3.46 and Algorithm 3.2, p. 171 (PDF page index 196).
- **Caption:** "Inverse warping algorithm: (a) a pixel g(x') is sampled from its
  corresponding location x = ĥ(x') in image f(x); (b) detail of the source and
  destination pixel locations." Algorithm 3.2 (the accompanying pseudocode) is included
  in the same crop.
- **Original source:** Szeliski's own diagram (not itself attributed to a third party in
  the caption).

### `szeliski_fig3_49_mesh_warping_alternatives.png`
- **Book figure:** Figure 3.49, p. 175 (PDF page index 200).
- **Caption:** "Image warping alternatives (Gomes, Darsa et al. 1999) © 1999 Morgan
  Kaufmann: (a) sparse control points → deformation grid; (b) denser set of control
  point correspondences; (c) oriented line correspondences; (d) uniform quadrilateral
  grid."
- **Original source, per the book's own citation:** Gomes, Darsa et al. (1999), ©
  1999 Morgan Kaufmann. Reproduced in the textbook and re-extracted here for the same
  educational purpose; the original copyright holder is Morgan Kaufmann.
