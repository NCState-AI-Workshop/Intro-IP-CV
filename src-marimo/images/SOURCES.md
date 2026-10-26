# Image sources and licensing

This folder holds shared raster assets for the `src-marimo` notebooks. Two kinds of
image live here, with different provenance:

- The five **standard test images** at the top level (`astronaut.png`, `coffee.png`,
  `chelsea.png`, `raccoon.png`, `coins.png`) — reusable across notebooks (color spaces,
  filtering, histogram equalization, ...).
- The **textbook figure extracts** in `textbook_figures/` — used by
  `07_sensors_and_color.py`, `11_multiresolution_representations.py`,
  `12_geometric_transformations.py`, `13_feature_detection.py`,
  `14_feature_matching_and_tracking.py`, `15_edge_detection.py`,
  `16_active_contours_and_segmentation.py`, and
  `17_lines_hough_and_ransac.py`, cropped from the course PDF for citation purposes.
- The **externally-sourced figures** in `external_figures/` — used by
  `18_optical_flow.py` for material beyond the textbook (an application image, and
  real video frame pairs from a standard research dataset), each cited below with
  its original source and license.

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

### `coins.png`
- **Function:** `skimage.data.coins()`
- **Subject:** Several coins of a few distinct sizes on a plain background — grayscale,
  replicated across 3 channels for consistency with the RGB loader used elsewhere.
  Chosen for its clean, well-separated circular boundaries (Hough circle detection,
  active-contour/snake initialization, watershed-style segmentation).
- **License:** Released under a free-of-restriction license by scikit-image; original
  images are in the public domain (no known copyright restrictions), per
  scikit-image's own data-source documentation.
- **Dimensions:** 303 × 384, originally grayscale.

Why these and not the classic "Lena" test image: Lena is a 1972 Playboy centerfold crop
whose continued use in course materials and papers has drawn increasing (and
well-founded) criticism; the images above give the same "standard test image" role with
clean, unambiguous public-domain/CC0 licensing.

## Textbook figure extracts (`textbook_figures/`)

Cropped directly from `Szeliski_CVAABook_2ndEd.pdf` (Richard Szeliski, *Computer Vision:
Algorithms and Applications*, 2nd ed., final draft Sept. 2021, Springer) using PyMuPDF,
for citation in `07_sensors_and_color.py`, `11_multiresolution_representations.py`,
`12_geometric_transformations.py`, `13_feature_detection.py`,
`14_feature_matching_and_tracking.py`, and `15_edge_detection.py`. Reproduced here
for educational, non-commercial classroom use with full attribution, consistent
with the book's own citation of its sources.

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

### `szeliski_fig7_4_aperture_problem.png`
- **Book figure:** Figure 7.4, p. 422 (PDF page index 447).
- **Caption:** "Aperture problems for different image patches: (a) stable
  ("corner-like") flow; (b) classic aperture problem (barber-pole illusion); (c)
  textureless region."
- **Original source:** Szeliski's own diagram (not itself attributed to a third party
  in the caption).

### `szeliski_fig7_5_autocorrelation_surfaces.png`
- **Book figure:** Figure 7.5, p. 423 (PDF page index 448).
- **Caption:** "Three auto-correlation surfaces $E_{AC}(\Delta u)$ shown as both
  grayscale images and surface plots: (a) the original image is marked with three
  red crosses to denote where the auto-correlation surfaces were computed; (b) this
  patch is from the flower bed (good unique minimum); (c) this patch is from the
  roof edge (one-dimensional aperture problem); and (d) this patch is from the cloud
  (no good peak)."
- **Original source:** Szeliski's own figure (not itself attributed to a third party
  in the caption).

### `szeliski_fig7_6_uncertainty_ellipse.png`
- **Book figure:** Figure 7.6, p. 425 (PDF page index 450).
- **Caption:** "Uncertainty ellipse corresponding to an eigenvalue analysis of the
  auto-correlation matrix A."
- **Original source:** Szeliski's own diagram (not itself attributed to a third party
  in the caption).

### `szeliski_fig7_11_dog_scale_space.png`
- **Book figure:** Figure 7.11, p. 430 (PDF page index 455).
- **Caption:** "Scale-space feature detection using a sub-octave Difference of
  Gaussian pyramid (Lowe 2004) © 2004 Springer: (a) Adjacent levels of a sub-octave
  Gaussian pyramid are subtracted to produce Difference of Gaussian images; (b)
  extrema (maxima and minima) in the resulting 3D volume are detected by comparing a
  pixel to its 26 neighbors."
- **Original source, per the book's own citation:** Lowe (2004), © 2004 Springer.
  Reproduced in the textbook and re-extracted here for the same educational purpose;
  the original copyright holder is Springer.

### `szeliski_fig7_12_orientation_histogram.png`
- **Book figure:** Figure 7.12, p. 431 (PDF page index 456).
- **Caption:** "A dominant orientation estimate can be computed by creating a
  histogram of all the gradient orientations (weighted by their magnitudes or after
  thresholding out small gradients) and then finding the significant peaks in this
  distribution (Lowe 2004) © 2004 Springer."
- **Original source, per the book's own citation:** Lowe (2004), © 2004 Springer.
  Reproduced in the textbook and re-extracted here for the same educational purpose;
  the original copyright holder is Springer.

### `szeliski_fig7_21_false_positives_negatives.png`
- **Book figure/table:** Figure 7.21 and Table 7.1, p. 442 (PDF page index 467).
- **Caption:** "False positives and negatives: The black digits 1 and 2 are features
  being matched against a database of features in other images..." Table 7.1 (the
  accompanying worked TP/FP/FN/TN/TPR/FPR/PPV/ACC example) is included in the same
  crop.
- **Original source:** Szeliski's own diagram/table (not itself attributed to a
  third party in the caption).

### `szeliski_fig7_23_nndr_matching.png`
- **Book figure:** Figure 7.23, p. 445 (PDF page index 470).
- **Caption:** "Fixed threshold, nearest neighbor, and nearest neighbor distance
  ratio matching. At a fixed distance threshold (dashed circles), descriptor D_A
  fails to match D_B and D_D incorrectly matches D_C and D_E..."
- **Original source:** Szeliski's own diagram (not itself attributed to a third
  party in the caption).

### `szeliski_fig7_32_human_boundary_detection.png`
- **Book figure:** Figure 7.32, p. 456 (PDF page index 481).
- **Caption:** "Human boundary detection (Martin, Fowlkes, and Malik 2004) © 2004
  IEEE. The darkness of the edges corresponds to how many human subjects marked an
  object boundary at that location."
- **Original source, per the book's own citation:** Martin, Fowlkes, and Malik
  (2004), © 2004 IEEE. Reproduced in the textbook and re-extracted here for the same
  educational purpose; the original copyright holder is IEEE.

## Externally-sourced figures (`external_figures/`)

Images used in `18_optical_flow.py`, `22_stereo_vision_and_dense_depth.py`,
and `24_nerf_and_gaussian_splatting.py` that come from outside the course
PDF, each with its own license and
attribution.

### `piv_vortex_pair.jpg`
- **Subject:** Particle Image Velocimetry (PIV) analysis of a Hamel–Oseen vortex
  pair — a textured field of tracer particles (seeded into a fluid) with the
  computed velocity vector field overlaid, including an inset showing the
  increase in spatial resolution from a multi-pass window-deformation technique.
  Used in notebook 18's applications intro as an example of optical-flow-style
  motion estimation in scientific/fluid imaging, beyond the book's own listed
  applications.
- **Source:** Wikimedia Commons, [`File:PIVlab_multipass.jpg`](https://commons.wikimedia.org/wiki/File:PIVlab_multipass.jpg).
- **Author:** Willa.
- **License:** CC BY-SA 3.0 (Creative Commons Attribution-ShareAlike 3.0
  Unported). Reproduced unmodified (resized only).

### `sintel_desert_frame1.png` / `sintel_desert_frame2.png`
### `sintel_overlook_frame1.png` / `sintel_overlook_frame2.png`
- **Subject:** Two consecutive-frame pairs (frames 8500–8501 and 7200–7201) from
  the Blender Foundation's open movie *Sintel*, used in notebook 18 §5 as real
  video frame pairs for dense optical flow estimation — the same movie the
  course textbook cites as the source of the modern MPI-Sintel optical flow
  benchmark (Butler, Wulff, Stanley, and Black, "A naturalistic open source
  movie for optical flow evaluation," ECCV 2012), per Szeliski §9.3's own
  recommendation that "more recent publications tend to focus... on the MPI
  Sintel dataset." Both were chosen specifically for having a panning camera
  over a richly-textured scene, which gives a clean, coherent (motion-parallax)
  flow field for illustration purposes.
- **Source:** Official Xiph.org Test Media collection,
  <https://media.xiph.org/sintel/sintel-1080-png/> (frame-exact PNG export of
  the finished film), frame numbers 8500/8501 and 7200/7201, resized down from
  1920×818 to 320×136 for this notebook.
- **Original creator:** Blender Foundation / Durian Film Project
  (<https://durian.blender.org>), directed by Colin Levy.
- **License:** CC BY 3.0 (Creative Commons Attribution 3.0 Unported), per the
  project's own README (<https://media.xiph.org/sintel/README.txt>): "Some
  Rights Reserved. Creative Commons Attribution 3.0 license," excepting logos
  included in the film (not present in the cropped frames used here).

### `sintel_desert_clip.mp4` / `sintel_overlook_clip.mp4`
- **Subject:** Short (32-frame, ~1.3s at the film's native 24fps) motion clips
  spanning the same two scenes as the still-frame pairs above, used in notebook
  18 §5 so students can watch the real motion directly, as a side-by-side
  comparison against the single-frame-pair optical flow estimate.
- **Source:** Same Official Xiph.org Test Media collection as above
  (<https://media.xiph.org/sintel/sintel-1080-png/>), frame ranges 8485–8516
  and 7185–7216, each frame resized to 320×136 and re-encoded to H.264/MP4
  with `ffmpeg` at the source framerate.
- **Original creator / License:** Same as the still frames above — Blender
  Foundation / Durian Film Project, CC BY 3.0.

### `middlebury_tsukuba_left.png` / `middlebury_tsukuba_right.png` / `middlebury_tsukuba_gt.npy`
### `middlebury_map_left.png` / `middlebury_map_right.png` / `middlebury_map_gt.npy`
### `middlebury_venus_left.png` / `middlebury_venus_right.png` / `middlebury_venus_gt.npy`
- **Subject:** Three classic rectified stereo pairs with pixel-accurate
  ground-truth disparity maps, used in notebook 22 as real, standard-dataset
  examples of block matching and dynamic-programming stereo correspondence,
  alongside that notebook's synthetic demo. Tsukuba (384×288, desk scene with
  camera/lamp/bust) uses the standard adjacent-column pair (`scene1.row3.col3`
  as left, `scene1.row3.col4` as right) from the multi-baseline sequence, with
  ground truth `truedisp.row3.col3.pgm` scaled by a factor of 16 (confirmed
  empirically: raw values are exact multiples of 16, giving a true disparity
  range of 0-14px). Map (284×216, tilted topographic relief map, `im0`/`im1`)
  and Venus (434×383, slanted poster scene, `im2`/`im6` of a 9-view sequence)
  both use ground truth scaled by a factor of 8, per the dataset's own
  documentation -- also confirmed empirically here by warping the right image
  by the recovered disparity and checking it reconstructs the left image
  (median reconstruction error 1.7, 6.0, and 1.2 (of 255) for Tsukuba, Map,
  and Venus respectively). The `_gt.npy` files store the already-descaled,
  true pixel-disparity arrays (float32) directly, so the notebook never needs
  to re-derive or guess the scale factor.
- **Source:** Middlebury Stereo Datasets, 2001 scene set,
  <https://vision.middlebury.edu/stereo/data/scenes2001/data/> (`tsukuba.zip`,
  `map.zip`, `venus.zip`). Tsukuba courtesy of the University of Tsukuba; Map
  courtesy of R. Szeliski and R. Zabih.
- **Citation:** D. Scharstein and R. Szeliski, "A Taxonomy and Evaluation of
  Dense Two-Frame Stereo Correspondence Algorithms," *International Journal
  of Computer Vision* 47(1-3):7-42, April-June 2002 -- the same taxonomy
  (matching cost, aggregation, optimization, refinement) notebook 22 itself
  follows, per Szeliski §12's own framing.
- **License:** The Middlebury Stereo Vision page states: "We grant permission
  to use and publish all images and disparity maps on this website," subject
  to citing the paper above -- the same kind of permission-plus-attribution
  license already relied on for the PIV photo and Sintel frames above. The
  Tsukuba pair is also reused directly (no new asset) in notebook 22 §2's
  before/after epipolar-rectification demo: each image is re-warped there by
  its own synthetic rotation-only homography to stand in for an unrectified
  capture, with Tsukuba's own ground-truth disparity giving exact-match
  points for the epipolar-line construction -- chosen over sourcing a second
  real, genuinely-unrectified stereo rig (attempted once via OpenCV's
  calibration sample photos and from-scratch pose recovery, which produced
  an implausibly large inter-camera rotation on cross-checking and wasn't
  reliable enough to present as ground truth).

### `nerf_fig1_pipeline_teaser.png`, `nerf_fig2_pipeline_detail.png`, `nerf_fig4_ablation.png`
- **Subject:** Three figures from the original NeRF paper, used in notebook
  24's overview of neural radiance fields. `nerf_fig1_pipeline_teaser.png` is
  Figure 1 (input images -> optimized NeRF -> rendered novel views, on the
  synthetic Drums scene). `nerf_fig2_pipeline_detail.png` is Figure 2 (the
  4-panel 5D-input / MLP-output / volume-rendering / rendering-loss
  pipeline). `nerf_fig4_ablation.png` is Figure 4 (ground truth vs. the full
  model vs. removing view-dependence vs. removing positional encoding, on
  the synthetic Lego bulldozer scene) -- the ablation that directly motivates
  §5's positional-encoding and view-dependence discussion. Each image is
  cropped directly from the paper's own figure region (PyMuPDF), with the
  original figure's own caption text excluded in favor of the notebook's own
  citation, then downscaled 2x for file size.
- **Source:** B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R.
  Ramamoorthi, and R. Ng, "NeRF: Representing Scenes as Neural Radiance
  Fields for View Synthesis," *ECCV 2020*. arXiv:2003.08934.
  <https://arxiv.org/abs/2003.08934>.
- **License:** Preprint freely available on arXiv; figures reproduced here
  for non-commercial educational/classroom commentary with full attribution
  to the original authors and venue, the same convention this course already
  uses for the `textbook_figures/` extracts (several of which are themselves
  third-party figures the Szeliski book reproduces under its own "© year
  Publisher" notices, e.g. its own reproduction of this very paper's Fig.
  14.24b).

### `gsplat_fig1_teaser_comparison.png`, `gsplat_fig2_pipeline.png`, `gsplat_fig3_shrunken_gaussians.png`, `gsplat_fig4_densification.png`
- **Subject:** Four figures from the original 3D Gaussian Splatting paper,
  used in notebook 24's overview. `gsplat_fig1_teaser_comparison.png` is (a
  crop of) Figure 1's quality/speed/training-time comparison against
  InstantNGP, Plenoxels, and Mip-NeRF360 on the Bicycle scene.
  `gsplat_fig2_pipeline.png` is Figure 2 (SfM points -> initialization -> 3D
  Gaussians -> projection -> differentiable tile rasterizer -> image, with
  the adaptive-density-control feedback loop) -- directly parallel to the
  NeRF pipeline figure above, and the figure that makes this method's
  SfM-point-cloud starting point explicit (the same bundle-adjustment output
  notebook 23 produces). `gsplat_fig3_shrunken_gaussians.png` is Figure 3
  (a rendered patch next to its own Gaussians, shrunk 60%, revealing their
  anisotropic shapes). `gsplat_fig4_densification.png` is Figure 4 (the
  clone/split adaptive densification scheme for under- vs.
  over-reconstructed regions). Cropped and downscaled the same way as the
  NeRF figures above.
- **Source:** B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, "3D
  Gaussian Splatting for Real-Time Radiance Field Rendering," *ACM
  Transactions on Graphics* 42(4), Article 1, August 2023. arXiv:2308.04079.
  <https://arxiv.org/abs/2308.04079>.
- **License:** "© 2023 Copyright held by the owner/author(s). Publication
  rights licensed to ACM" (per the paper's own copyright line -- the authors
  retain copyright). Figures reproduced here for non-commercial educational
  classroom commentary with full attribution, the same convention as the
  NeRF figures above. This paper postdates Szeliski's 2nd-edition draft
  (Sept. 2021) by nearly two years, so it appears nowhere in the book
  notebooks 1-23 have been citing all along -- the first notebook in this
  course to source a paper the textbook itself has no knowledge of.
