# Reconstruction of 3D curve based on two 2D projections

- Python code to reconstruct a three-dimensional space curve from two orthogonal two-dimensional projections. 
- This code has been used in reference [1] to digitalize three-dimensional shapes of a beating cilium 
based on original stereoscopic high-speed video-microscopy by Machemer et al.
  
- Final results (3D coordinates of shapes) can be found in `res/pipe_final/`.
- Additionally, in `shapes_final_manuscript/shapes_data` are the shapes, originally obtained for the reference [1]. 
  (The shapes slightly differ (`dx/L < 0.03`) from the ones presented  in  `res/pipe_final/`
   because of different package versions used).

### Version requirements

```
scipy=1.5.3
numpy=1.19.4
matplotlib=3.3.3
vedo=2020.4.2 # only for 3D visualization
vtk=8.2.0     # only for 3D visualization
```


### How-to: optimization with `scipy`

- Tutorial https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
- Documentation https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize
- How to choose a method https://scipy-lectures.org/advanced/mathematical_optimization/index.html


### Contents

#### Data set

- Time sequences of three-dimensional shapes of beating cilia on the surface of unicellular Paramecium had been recorded
  by Machemer using high-speed video-microscopy in with stereoscopic recording of orthogonal two-dimensional projections in 1972;
  this cilia beat pattern was subsequently presented in Figure 2 of Naitoh et al. 1984.
- We manually digitalized this historic data set by manual tracking
  (using manual tracking in Adobe Illustrator and export as svg-files).
  In folder `data/raw/`, files `x0-#.dat`, `y0-#.dat` describe one projection of a shape,
   `x20-#.dat`, `z0-#.dat` describe the second orthogonal projection.
- We found a mismatch between scale of projections
  (i.e., $\max{x_j}-\min{x_j} \neq \max{x^{(2)}_j}-\min{x^{(2)}_j}$
  We rescaled the second projection by a constant factor. Results are in `data/rescale/`. 
  (For reference, the original matlab scripts are present in `data/`.)
  

#### Reusable code

`curve3d.py`
- construct 3d curve by defining segments
- reconstruct segments information from curve points
- Description of the coordinate system used.

#### Examples
`ex00_optimize.py`
 - introduce scipy optimization tools

`ex01_optimize_curve.py`
- test reconstruction procedure on example curves
    
#### Pipeline steps

`pipe00_rescale_v4.py`
- Reconstruct 3D coordinates via optimization procedure to ensure smoothness of the curve.
  Optimization have 3 terms: "points", "tangents", "smoothness".
- Regularization weights are dimensionless. 
- Started with `smooth_weigh=0`, smooth_weight=0` 
  (optimize only for alignment of tangent vectors). With visual inspection found the best points_weight.
  - When `points_weight` is too small; the curves will be parallel, but points will not necessarily align.
  - When `points_weight` is too large (and no smoothing) - projection points oscillate around the target curve.
- Increased `smooth_weight` until smoothing starts to mask essential features of the space curve.
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


#### Visualizations

- `vis00_calc_length.py` visualize shape lengths after `pipe00`

- `vis01a_plot_shapes_pipe00.py` 
  `vis01b_plot_shapes_final.py`
  `vis01b_plot_shapes_pipe00_and_final.py`
   plot shapes after different steps

- `vis02_plot_angle_kymograph.py`
   kymograph of angles $\theta$, $\psi$

- `vis03_plot_tangent_product_kymograph.py`
    kymograph of tangent dot product
  
### Authors

- [Anton Solovev](https://github.com/icemtel)
- [Benjamin M. Friedrich](https://cfaed.tu-dresden.de/friedrich-home) benjamin.m.friedrich@tu-dresden.de

Publication to cite: [1]

- [1]: [Solovev & Friedrich 2020 EPJ E ST](https://link.springer.com/article/10.1140/epje/s10189-021-00016-x);  also available on [arXiv](https://arxiv.org/abs/2010.08111 ) 
- [2]: [Solovev & Friedrich 2020b arXiv](https://arxiv.org/abs/2012.11741)
