

% ============================================================
% Load raw trace
% ============================================================
raw_data = load( ...
    '/scratch/dazhang/MATLAB/EC135_513/ls_trace_EC135_Channel_303_binsize_1_Block_1.mat');

raw_trace = raw_data.ls_traces(257, :);


% ============================================================
% Load filtered trace
% ============================================================
filtered_data = load( ...
    '/scratch/dazhang/MATLAB/EC135_513/ls_filtered_trace_EC135_Channel_303_binsize_1_Block_1.mat');

filtered_trace = filtered_data.ls_filtered_traces(257, :);


% ============================================================
% Parameters
% ============================================================
left_boundary  = 205;
right_boundary = 307;

x_margin = 10;

xmin = left_boundary  - x_margin;
xmax = right_boundary + x_margin;


% ============================================================
% Display ranges (slightly larger than labeled scale)
% ============================================================

% Raw trace display range
raw_ymin = -1300;
raw_ymax =  1300;

% Filtered trace display range
filtered_ymin = -130;
filtered_ymax =  130;


% ============================================================
% Scale values to mark
% ============================================================

% Raw scale labels / ticks
raw_tick_max =  1000;
raw_tick_min = -1000;

% Filtered scale labels / ticks
filtered_tick_max =  100;
filtered_tick_min = -100;


% ============================================================
% Normalized Y positions for labels
% ============================================================
raw_tick_max_norm = (raw_tick_max - raw_ymin) / (raw_ymax - raw_ymin);
raw_tick_min_norm = (raw_tick_min - raw_ymin) / (raw_ymax - raw_ymin);
filtered_tick_max_norm = (filtered_tick_max - filtered_ymin) / (filtered_ymax - filtered_ymin);
filtered_tick_min_norm = (filtered_tick_min - filtered_ymin) / (filtered_ymax - filtered_ymin);


% ============================================================
% Text positions
% ============================================================
left_label_x  = -0.05;
right_label_x =  1.05;


% ============================================================
% Colors
% ============================================================
raw_color = [0 0 1];            % blue
filtered_color = [1 0.5 0];  % orange


fig = figure('Color', 'w');
% AXIS 1: RAW TRACE

ax1 = axes(fig);
hold(ax1, 'on');

plot(ax1, 1:513, raw_trace, 'Color', raw_color, 'LineWidth', 2);

xlim(ax1, [xmin xmax]);
ylim(ax1, [raw_ymin raw_ymax]);

ax1.TickLength = [0 0];
% ax1.XTickLabel = [];
% ax1.YTickLabel = [];
ax1.XGrid = 'on';
ax1.YGrid = 'on';
ax1.XMinorTick = 'on';
ax1.YMinorTick = 'on';
ax1.XGrid = 'on'; %not minor!
ax1.YGrid = 'on';
ax1.MinorGridLineStyle = ':';
ax1.MinorGridColor = [0.7 0.7 0.7];
ax1.MinorGridAlpha = 0.18;
ax1.YAxis.TickLabelGapOffset=25;
ax1.XAxis.TickLabelGapOffset=6;



% ============================================================
% Dashed boundaries
% ============================================================
xline(ax1, left_boundary, '--', 'Color', [0.5 0.5 0.5],'LineWidth', 2);
xline(ax1, right_boundary, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 2);


% ============================================================
% Raw axis appearance
% ============================================================
ax1.Color = 'w';
% ax1.XTick = [];
% ax1.YTick = [];
ax1.XTickLabel=[];
ax1.Box = 'on';
ax1.LineWidth = 1.8;
ax1.XColor = 'k';
ax1.YColor = 'k';
ax1.FontSize = 16;


% ============================================================
% AXIS 2: FILTERED TRACE
% ============================================================
ax2 = axes(fig, 'Position', ax1.Position, 'Color', 'none');
hold(ax2, 'on');
plot(ax2, 1:513, filtered_trace, 'Color', filtered_color, 'LineWidth', 2);

xlim(ax2, [xmin xmax]);
ylim(ax2, [filtered_ymin filtered_ymax]);


% ============================================================
% Filtered axis appearance
% ============================================================
ax2.XTick = [];
ax2.XTickLabel = [];
ax2.YAxisLocation = 'right';
ax2.YTick = [-100 100];
ax2.YTickLabel = {'-100' '100'};

ax2.Box = 'off';
ax2.XColor = 'none';
ax2.YColor = 'k';
ax2.Color = 'none';
ax2.YColor = 'k';
ax2.FontSize = 22;
ax2.YAxis.TickLabelGapOffset=25;
ax2.XAxis.TickLabelGapOffset=6;





% ============================================================
% Link X axes
% ============================================================
linkaxes([ax1 ax2], 'x');


% ============================================================
% Solid scale ticks (all black)
% ============================================================
tick_length = 4;

% LEFT: +1200
plot(ax1, ...
    [xmin - tick_length, xmin], ...
    [raw_tick_max, raw_tick_max], ...
    'Color', 'k', ...
    'LineStyle', '-', ...
    'LineWidth', 1.8, ...
    'Clipping', 'off');

% LEFT: -1200
plot(ax1, ...
    [xmin - tick_length, xmin], ...
    [raw_tick_min, raw_tick_min], ...
    'Color', 'k', ...
    'LineStyle', '-', ...
    'LineWidth', 1.8, ...
    'Clipping', 'off');

% RIGHT: +120
plot(ax2, ...
    [xmax, xmax + tick_length], ...
    [filtered_tick_max, filtered_tick_max], ...
    'Color', 'k', ...
    'LineStyle', '-', ...
    'LineWidth', 1.8, ...
    'Clipping', 'off');

% RIGHT: -120
plot(ax2, ...
    [xmax, xmax + tick_length], ...
    [filtered_tick_min, filtered_tick_min], ...
    'Color', 'k', ...
    'LineStyle', '-', ...
    'LineWidth', 1.8, ...
    'Clipping', 'off');


uistack(ax1, 'bottom');
uistack(ax2, 'top');

hold(ax1, 'off');
hold(ax2, 'off');