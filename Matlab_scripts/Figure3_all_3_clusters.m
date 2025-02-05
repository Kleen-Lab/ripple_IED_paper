close all
array_name = load('~/Desktop/Avg_cluster_0_.mat').array_name;
figure;
colormap(parula);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% c = colorbar;
caxis([-.55 .55]);
% c.FontSize = 24; % Set larger font size for the color bar text
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
% set(gca, 'Ylim', [1 200]);
% set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gca, 'YTick', []);
array_name = load('~/Desktop/Avg_cluster_1_.mat').array_name;
figure;
colormap(parula);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
ax = gca;               % Get current axes

ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;
% c = colorbar;
caxis([-.55 .55]);
% c.FontSize = 24; % Set larger font size for the color bar text
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
% set(gca, 'Ylim', [1 200]);
% set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gca, 'YTick', []);
array_name = load('~/Desktop/Avg_cluster_2_.mat').array_name;
figure;
colormap(parula);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% c = colorbar;
caxis([-.55 .55]);
% c.FontSize = 24; % Set larger font size for the color bar text
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
% set(gca, 'Ylim', [1 200]);
% set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gca, 'YTick', []);
array_name = load('~/Desktop/Avg_cluster_3_.mat').array_name;
figure;
colormap(parula);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% c = colorbar;
caxis([-.55 .55]);
% c.FontSize = 24; % Set larger font size for the color bar text
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
% set(gca, 'Ylim', [1 200]);
% set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gca, 'YTick', []);
array_name = load('~/Desktop/Avg_cluster_4_.mat').array_name;
figure;
colormap(parula);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% c = colorbar;
caxis([-.55 .55]);
% c.FontSize = 24; % Set larger font size for the color bar text
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
% set(gca, 'Ylim', [1 200]);
% set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gca, 'YTick', []);
