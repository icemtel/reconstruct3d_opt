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
k=nan(num_frames,1);
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

% Don't skip frames - just see what's the scaling factor is
% Identify problematic frames, where y(s) is not monotonic
%y_diff = diff(y_side);
%
% if any(y_diff >= 0) && any(y_diff < 0)
%     fprintf("%d\n", frame_ids(iframe))
%     continue  % TODO: Make a solution for these frames
% end

% top : 23-35; side 23-37?!

% Scaling factor
k(iframe) = max(abs(y_side))/max(abs(y_top));
fprintf("%d    %f\n", frame_ids(iframe), k(iframe))

% Rescale the second projection
y_top = y_top * k(iframe);
x_top = x_top * k(iframe);

% Interpolate for regular y-spacing
y_shifted_side = y_side - y_side(1);
y_shifted_top = y_top - y_top(1);
ymax = max(y_shifted_side);


% % - interpolate
% dy = 1;
% y_shifted0 = 0:dy:(maxpoints-1)*dy; % 
% ind=find(y_shifted0<ymax);
% % side
% coords(iframe,ind,1) = interp1(y_shifted_side,y_side,y_shifted0(ind),'pchip');
% coords(iframe,ind,2) = interp1(y_shifted_side,z_side,y_shifted0(ind),'pchip');
% % top
% coords(iframe,ind,3) = interp1(y_shifted_top,y_top,y_shifted0(ind),'pchip');
% coords(iframe,ind,4) = interp1(y_shifted_top,x_top,y_shifted0(ind),'pchip');


end %iframe

%% Visualize
figure(1), clf, hold on

% good_frames = find(k(isfinite(k)));
% for iframe = transpose(good_frames)
%     plot(coords(iframe,:,1),coords(iframe,:,2),'color',hsv2rgb([iframe/(num_frames+1) 1 1]))
%     lastpoint=sum(isfinite(coords(iframe,:,1))); % lol
%    text(coords(iframe,lastpoint,1)+5,coords(iframe,lastpoint,2),sprintf('%d',iframe),'color',hsv2rgb([iframe/(num_frames+1) 1 1]))
% end
% daspect([1 1 1])

