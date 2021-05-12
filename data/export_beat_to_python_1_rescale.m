% Load and process flagellar beat patterns tracked by hand
clear all

% frames 1-41, but not every frame was digitized
num_frames = 21; 
frame_ids = [1:2:17 18 21:2:41];


% set reference point for proximal tip by hand
% Side view (xy) TODO: change to yz?
x0=117; % adjust manually for each new beat pattern
y0=179.38;

% Top view
x0_top = 119.35;
y0_top = 316.06;

 % Later rescale top projection
rescale_factor = 1.12;


figure(1), clf, hold on

% loop over frames; Side View
for iframe=1:num_frames

iframe_raw=frame_ids(iframe);
% Side view
fname = sprintf('digitize/%d.dat',iframe_raw); % If start from 0: %02d
data = load(fname);

ndata = length(data);
npoints_raw = ndata / 2;

xflag_raw = data(1:2:end);
yflag_raw = data(2:2:end);  

xflag_raw= xflag_raw-x0;
yflag_raw= -yflag_raw + y0; % Reversed y because of svg coordinate system


% reverse shapes
xflag_raw=xflag_raw(end:-1:1);
yflag_raw=yflag_raw(end:-1:1);

% add proximal tip --------------------------------------------------------
xflag_raw=[0 xflag_raw];
yflag_raw=[0 yflag_raw];

% Plot raw shapes
% plot(xflag_raw,yflag_raw,'-','color',hsv2rgb([(iframe-1)/num_frames 1 1]))
% text(xflag_raw(npoints_raw)+5,yflag_raw(npoints_raw),sprintf('%d',iframe),'color',hsv2rgb([(iframe-1)/num_frames 1 1]))
% daspect([1 1 1])

% Save shapes
xname = sprintf('res/rescale/x0-%d.dat',iframe_raw);
yname = sprintf('res/rescale/y0-%d.dat',iframe_raw);
save(xname,'xflag_raw','-ASCII');
save(yname,'yflag_raw','-ASCII');
end % iframe

% loop over frames; Top View
for iframe=1:num_frames

iframe_raw=frame_ids(iframe);
% Side view
fname = sprintf('digitize/%d-top.dat',iframe_raw); % If start from 0: %02d
data = load(fname);

ndata = length(data);
npoints_raw = ndata / 2;


xflag_raw = data(1:2:end);
yflag_raw = data(2:2:end);  

xflag_raw= xflag_raw-x0_top;
yflag_raw= -yflag_raw + y0_top; % Reversed y because of svg coordinate system


% reverse shapes
xflag_raw=xflag_raw(end:-1:1);
yflag_raw=yflag_raw(end:-1:1);

% add proximal tip --------------------------------------------------------
xflag_raw=[0 xflag_raw];
yflag_raw=[0 yflag_raw];

% Plot raw shapes
%plot(xflag_raw,yflag_raw,'-','color',hsv2rgb([(iframe-1)/num_frames 1 1]))
%text(xflag_raw(npoints_raw)+5,yflag_raw(npoints_raw),sprintf('%d',iframe),'color',hsv2rgb([(iframe-1)/num_frames 1 1]))
%daspect([1 1 1])


% 1: Rescale shapes with constant factor
xflag = rescale_factor * xflag_raw;
yflag = rescale_factor * yflag_raw;
% Save shapes
x2name = sprintf('res/rescale/x20-%d.dat',iframe_raw);
zname = sprintf('res/rescale/z0-%d.dat',iframe_raw);
save(x2name,'xflag','-ASCII');
save(zname,'yflag','-ASCII');
end % iframe

return


%% Rescale
L=17; % [um] target length of cilium: TODO: doublecheck



