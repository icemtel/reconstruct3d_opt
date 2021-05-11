# Reconstruction of 3D curve based on two 2D projections

Python code to reconstruct a three-dimensional space curve from two orthogonal two-dimensional projections. 
This code has been used in reference [1] to digitalize three-dimensional shapes of a beating cilium 
based on original stereoscopic high-speed video-microscopy by Machemer et al.

### Version requirements

```
scipy>=1.2.1
matplotlib>=3.0
vedo>=2020.4 # only for 3D visualization
```

### External links

Optimization with SciPy
- Tutorial https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
- Documentation https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize
- How to choose a method https://scipy-lectures.org/advanced/mathematical_optimization/index.html

### Data set
- Time sequences of three-dimensional shapes of beating cilia on the surface of unicellular Paramecium had been recorded by Machemer using high-speed video-microscopy in with stereoscopic recording of orthogonal two-dimensional projections in 1972; this cilia beat pattern was subsequently presented in Figure 2 of Naitoh et al. 1984. We had manually digitalized this historic data set by manual tracking (using manual tracking in Adobe Illustrator and export as svg-files).
- The coordinate system used is described in `curve3d.py'

#### Choice of weights
- Regularization weights are dimensionless
- Start with smooth_weight = 0; With visual inspection find the best points_weight.
  - If points_weight is too small; the curves will be parallel, but points will not necessarily align.
  - If points_weight is too large (and no smoothing) - projection points might oscillate around the target curve.
- Increase smooth_weight until smoothing starts to mask essential features of the space curve.

### Scripts

#### Reusable code

`curve3d.py`

#### Examples
`ex00_optimize.py`
 - get used to scipy optimization tools
`curve3d.py`
 - construct 3d curve by defining segments
 - reconstruct segments information from curve points
`example_optimize_curve.py`
 - Try to apply optimization tools to fit 3d curve to projections (simple, e.g. a line)
 - Line example:
   - works good, when M=30 (and tol at least 10 ** -3); If M=20, then points visibly oscillate around the line
   - Good, even when initial guess is very far off
 - Helix - OK
 - Anisotropic helix ( a/b =27):
   - Not good precision with N,M = (30,15) points; (30,30)
   - Didn't finish in reasonable time with 40 points
   - Better initial conditions: fits xz (smaller projection) good, but in general fails
 - Anisotropic helix (a/b = 9): initial conditions - Helix + a lot of noise
   - Works almost fine; probably just need more control points
   - (60,60) points -> fits almost perfectly
 - Helix - different projections: x = cos(s)
   - Perfect fit, (30,30)
    
#### Pipeline steps

`pipe00_rescale_v4.py`
- reconstruct 3D coordinates; introduce optimization penalty for kinks -> ensure smoothness of the curve.
- Final params: `N = 60` `M = 60` `eps = 1e-3` `ds_eps = 1e-3` `points_weight = 100` `smooth_weight = 40`.
  [May use smaller N=M=30 to compute faster while testing]

`pipe01_combine_coords_in_single_file.py`
- store x coordinates for each shape in a single file. Same for y,z.

`pipe02_reinterpolate_drds.py`
In this step, 
- we set shapes to the same length 
- we represent curves as a number of segments of the same lengths, 
  represented by polar angles $(\psi, \theta)$ which set the direction (tangent) of each segment.

Details:
- After previous steps, shapes have different lengths.
  Here, the longer shapes are cut, and shorter are extrapolated.
  - Extrapolation is done in terms of components of tangent vectors. 
  - We assume that if $||t|| \neq 1 $, then it changes the length of the shape.
  - We redo the extrapolation/interpolation until all tangents have unit length.
- Target length = mean of the original shape lengths.
- Tangents are represented in terms of polar angles $(\psi, \theta)$.

`pipe03_fourier_of_angles.py`
- We fit a Fourier sum to each of angles $\psi$ , $\theta$ as function of phase $\varphi$.
- We keep the sum up to order 4 (four sine terms, four cosine terms, one constant).
  This preserves the main features (i.e. the distal tip velocity profile), yet provides some smoothing 
  and allows us to find shapes and velocities for any phase.

`pipe04_shift_phase_rescale_length_mirror.py`
- Shift phase, such that $\varphi \approx 0$ at the beginning of the recovery stroke.
- rescale (from px to micron)
- 
- FIX mirrored data; -> to have the correct 3D orientation, we mirror the shapes [see shapes in Naitoh1984]

`pipe05_get_normals.py`
- also extract direction of normal vectors (some normals; not the Frenet frame).


#### Visualization

- `vis00_calc_length.py` visualize shape lengths after `pipe00`

- `vis01a_plot_shapes_pipe00.py` 
  `vis01b_plot_shapes_final.py`
  `vis01b_plot_shapes_pipe00_and_final.py`
   plot shapes after different steps


#### Tests

- `test00_check_angle_reconstruction.py` check that Euclidean coordinates can be correctly reconstructed with local angle representation.

### Authors

- Anton Solovev anton.solovev@tu-dresden.de
- Benjamin M. Friedrich benjamin.m.friedrich@tu-dresden.de

Publication to cite: [1]

[1] [Solovev & Friedrich 2020 EPJ E ST](https://link.springer.com/article/10.1140/epje/s10189-021-00016-x);  also available on [arXiv](https://arxiv.org/abs/2010.08111 ) 
