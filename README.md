# Reconstruction of 3D curve based on two 2D projections

Code used to reconstruct cilium shapes, presented in [1]

### Requirements

```
scipy>=1.2.1
matplotlib>=3.0
vedo>=2020.4 # only for 3D visualization
```

### Literature

Optimization
- Tutorial https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
- Documentation https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize
- How to choose a method https://scipy-lectures.org/advanced/mathematical_optimization/index.html


### INFO
- Data set: Machemer 1974, Naitoh 1984 -> Fig.2; points were manually extracted via illustrator -> export to svg;
- Description of coordinate system in curve3d.py

### Choice of parameters 

- For testing: use 30 points - good value to test parameters. More points is helpful if curvature is locally big.

#### Choice of weights
- Weights are dimensionless.
- Start with smooth_weight = 0; With visual inspection find the best points_weight.
  - If points_weight is too small; the curves will be parallel, but points won't necessarily align.
  - If points_weight is too big (and no smoothing) - projection points might oscillate around the target curve.
- Increase smooth_weight until smoothing starts to destroy some features of the curve.

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
- Final params: N = 60 M = 60 eps = 10 ** -3 ds_eps = 10 ** -3 points_weight = 100 smooth_weight = 40

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

- `vis02_plot_angle_kymograph.py`
   kymograph of angles $\theta$, $\psi$

- `vis03_plot_tangent_product_kymograph.py`
    kymograph of tangent dot product
#### Tests

- `test00_check_angle_reconstruction.py` check that Euclidean coordinates can be correctly reconstructed with local angle representation.

### Authors

- Anton Solovev anton.solovev@tu-dresden.de
- Benjamin Friedrich benjamin.m.friedrich@tu-dresden.de

Publication to cite: [1]

[1] [Solovev & Friedrich 2020 EPJ E ST](https://link.springer.com/article/10.1140/epje/s10189-021-00016-x);  [pre-print](https://arxiv.org/abs/2010.08111 ) 