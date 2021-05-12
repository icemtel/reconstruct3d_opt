% Find scale factor between projections; for each shape
% 1 number - scale determined by abs max coordinate
% then 2 numbers - scale determined by min, and by max coordinate
clear all

num_frames = 21; 
frame_ids = [1:2:17 18 21:2:41]


% set reference point for proximal tip by hand
% Side view (yz)?
x0=117; % adjust manually for each new beat pattern
y0=179.38;

% Top view (yx)
x0_top = 119.35;
y0_top = 316.06;



maxpoints = 180; % large number; cut later
coords = nan(num_frames,maxpoints,4); % y_side z_side y_top x_top

% loop over frames
k_abs=nan(num_frames,1);
k_min=nan(num_frames,1);
k_max=nan(num_frames,1);
for iframe=1:num_frames

iframe_raw=frame_ids(iframe);

% Side view
fname = sprintf('digitize/%d.dat',iframe_raw); % If start from 0: %02d
data = load(fname);
% Top view
fname = sprintf('digitize/%d-top.dat',iframe_raw); % If start from 0: %02d
data_top = load(fname);


ndata = length(data);
npoints_raw = ndata / 2;


y_side = data(1:2:end);
z_side = data(2:2:end);  
y_top = data_top(1:2:end);
x_top = data_top(2:2:end);  


y_side= y_side-x0;
z_side= -z_side + y0; % Reversed y because of svg coordinate system
y_top=y_top-x0_top;
x_top=-x_top+y0_top;% Reversed y because of svg coordinate system

% reverse shapes
y_side=y_side(end:-1:1);
z_side=z_side(end:-1:1);
y_top=y_top(end:-1:1);
x_top=x_top(end:-1:1);


% Scaling factor
k_abs(iframe) = max(abs(y_side))/max(abs(y_top));
if abs(min(y_side)) >= 0.1 * abs(max(y_side)) && 10 * abs(max(y_side)) >= abs(min(y_side))% if too small - keep nan
    k_min(iframe) = min(y_side)/min(y_top); 
    k_max(iframe) = max(y_side)/max(y_top);
end



fprintf("%d    %f    %f    %f\n", frame_ids(iframe), k_abs(iframe), k_min(iframe), k_max(iframe))

end %iframe
