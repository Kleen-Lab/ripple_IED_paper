colorlimits = [0.0145 0.0175];
array_name = load('~/Desktop/avg_ripple_attention_psudo_all.mat').array_name;
figure;
subplot(2,2,1)
colormap(jet);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
set(gca, 'Ylim', [1 200]);
set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gcf, 'color','white')
caxis(colorlimits);
colorbar
caxis



array_name = load('~/Desktop/avg_spike_attention_psudo_all.mat').array_name;
subplot(2,2,2)
colormap(jet);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
set(gca, 'Ylim', [1 200]);
set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gcf, 'color','white')
caxis(colorlimits);
colorbar
caxis



array_name = load('~/Desktop/avg_ripple_attention_all.mat').array_name;
subplot(2,2,3)
colormap(jet);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
set(gca, 'Ylim', [1 200]);
set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gcf, 'color','white')
caxis(colorlimits);
colorbar
caxis






array_name = load('~/Desktop/avg_spike_attention_all.mat').array_name;
subplot(2,2,4)
colormap(jet);
imagesc(1:size(array_name,2), freqval, flipud(array_name));
set(gca, 'YDir', 'normal');
set(gca, 'YScale', 'log')
set(gca, 'Ylim', [1 200]);
set(gca,'YTick',([1 4 10 25 70 200]))
set(gca, 'XTick', []);
set(gcf, 'color','white')
caxis(colorlimits);
colorbar
caxis