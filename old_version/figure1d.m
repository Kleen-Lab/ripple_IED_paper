
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% First example %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time_start = 205-51;
time_end = 308+51;
x_ticks = [-0.2:0.1:0.2];
x_lims = [-0.2 0.2];
q_133 = load('/data/an/transformed/RR_h_referential_wSPKS/EC133/AN/EC133_AN_ch293_wSPKS.mat').q;

block_of_interest = 1;
candi_of_interest = 53;
freqval = q_133.frx;
ts=getts(size(q_133.Rdata(block_of_interest).Rspects, 2), q_133.sfx)-.5;
[raw_trace, filtered_trace, spectrogram] = plot_trace_filter_spec(q_133, block_of_interest, candi_of_interest);

figure;
f.Position = [10 10 900 600];

subplot(4,4,1);  
plot(ts(time_start:time_end), raw_trace(time_start:time_end), 'color', 'blue','LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% y_max = int32(max(abs(ylim)));
y_max = 300;
xlim(x_lims)
ylim([-y_max y_max]) % first example


yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end

set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
set(gca, 'YTickLabel', yTickLabels);
set(gca, 'xTickLabel','')



subplot(4,4,5); 
plot(ts(time_start:time_end), filtered_trace(time_start:time_end), 'color', [240 100 10]/256,'LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% y_max = int32(max(abs(ylim)));
y_max = 25;
xlim(x_lims)
ylim([-y_max y_max]) % first example
yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end

set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
set(gca, 'YTickLabel', yTickLabels);
set(gca, 'xTickLabel','')


subplot(4,4,[9 13]);   
pcolorjk(ts(time_start:time_end), q_133.frx(q_133.frx>=1), spectrogram(:,time_start:time_end));
set(gca,'yscale','log');
caxis([-6 6]);

yt=[2 4 10 25 70 200];
set(gca, "yTick", yt,'YTickLabel',cellstr(num2str(round(yt)'))')
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% set(gca, "YTickLabel",yTickLabels)

hold on; plot(xlim,[70 70],'w-', 'LineWidth',5);
hold on; plot(xlim,[150 150],'w-', 'LineWidth',5);
hold on; plot([0 0],ylim,'k-');






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Seconed example %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
block_of_interest = 1;
candi_of_interest = 21;
freqval = q_133.frx;
ts=getts(size(q_133.Rdata(block_of_interest).Rspects, 2), q_133.sfx)-.5;
[raw_trace, filtered_trace, spectrogram] = plot_trace_filter_spec(q_133, block_of_interest, candi_of_interest);




subplot(4,4,2); 
plot(ts(time_start:time_end), raw_trace(time_start:time_end), 'color', 'blue','LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% y_max = int32(max(abs(ylim)));
y_max = 250;
xlim(x_lims)
ylim([-y_max y_max]) % first example
yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end

set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)
% set(gca, 'XTick', [])
% set(gca, 'YTick', [])
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
% set(gca, 'YTickLabel', yTickLabels);
set(gca, 'YTickLabel', '');
set(gca, 'xTickLabel','')



subplot(4,4,6);  
plot(ts(time_start:time_end), filtered_trace(time_start:time_end), 'color', [240 100 10]/256,'LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
y_max = 25;
xlim(x_lims)
ylim([-y_max y_max]) % first example
yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end

set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
% set(gca, 'YTickLabel', yTickLabels);
set(gca, 'YTickLabel', '');
set(gca, 'xTickLabel','')




subplot(4,4,[10 14]); 
pcolorjk(ts(time_start:time_end), q_133.frx(q_133.frx>=1), spectrogram(:,time_start:time_end));
set(gca,'yscale','log');
caxis([-6 6]);

yt=[2 4 10 25 70 200];
% set(gca, "yTick", yt,'YTickLabel',cellstr(num2str(round(yt)'))')
set(gca, "yTick",[])
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% set(gca, "YTickLabel",yTickLabels)
hold on; plot(xlim,[70 70],'w-', 'LineWidth',5);
hold on; plot(xlim,[150 150],'w-', 'LineWidth',5);
hold on; plot([0 0],ylim,'k-');
set(gcf,"Color", 'white')







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Third example %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

block_of_interest = 1;
candi_of_interest = 8;
freqval = q_133.frx;
ts=getts(size(q_133.Rdata(block_of_interest).Rspects, 2), q_133.sfx)-.5;
[raw_trace, filtered_trace, spectrogram] = plot_trace_filter_spec(q_133, block_of_interest, candi_of_interest);



subplot(4,4,3); 
plot(ts(time_start:time_end), raw_trace(time_start:time_end), 'color', 'blue','LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
y_max = 250;
xlim(x_lims)
ylim([-y_max y_max]) % first example

yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end
% 
set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)
% set(gca, 'XTick', [])
% set(gca, 'YTick', [])
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
% set(gca, 'YTickLabel', yTickLabels);
set(gca, 'YTickLabel', '');
set(gca, 'xTickLabel','')



subplot(4,4,7); 
plot(ts(time_start:time_end), filtered_trace(time_start:time_end), 'color', [240 100 10]/256,'LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% y_max = int32(max(abs(ylim)));
y_max = 25;
xlim(x_lims)
ylim([-y_max y_max]) % first example
yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end

set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)
% set(gca, 'XTick', [])
% set(gca, 'YTick', [])
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
% set(gca, 'YTickLabel', yTickLabels);
set(gca, 'YTickLabel', '');
set(gca, 'xTickLabel','')


subplot(4,4,[11 15]);  
pcolorjk(ts(time_start:time_end), q_133.frx(q_133.frx>=1), spectrogram(:,time_start:time_end));
caxis([-6 6]);

set(gca,'yscale','log');
yt=[2 4 10 25 70 200];
% set(gca, "yTick", yt,'YTickLabel',cellstr(num2str(round(yt)'))')
set(gca, "yTick",[])
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% set(gca, "YTickLabel",yTickLabels)

hold on; plot(xlim,[70 70],'w-', 'LineWidth',5);
hold on; plot(xlim,[150 150],'w-', 'LineWidth',5);
hold on; plot([0 0],ylim,'k-');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Forth example %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
block_of_interest = 1;
candi_of_interest = 89;
freqval = q_133.frx;
ts=getts(size(q_133.Rdata(block_of_interest).Rspects, 2), q_133.sfx)-.5;


[raw_trace, filtered_trace, spectrogram] = plot_trace_filter_spec(q_133, block_of_interest, candi_of_interest);



subplot(4,4,4);   
plot(ts(time_start:time_end), raw_trace(time_start:time_end), 'color', 'blue','LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% y_max = max(abs(ylim));
y_max = 250;
xlim(x_lims)
ylim([-y_max y_max]) % first example


yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end
% 
set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)

% set(gca, 'XTick', [])
% set(gca, 'YTick', [])
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
% set(gca, 'YTickLabel', yTickLabels);
set(gca, 'YTickLabel', '');
set(gca, 'xTickLabel','')



subplot(4,4,8);   
plot(ts(time_start:time_end), filtered_trace(time_start:time_end), 'color', [240 100 10]/256,'LineWidth',3);
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% y_max = max(abs(ylim));
y_max = 25;
xlim(x_lims)
ylim([-y_max y_max]) % first example
yTickPositions = [-y_max:y_max/4:y_max];
yTickLabels = cell(1, length(yTickPositions)); % Initialize cell array for labels
for i = 1:length(yTickPositions)
    if yTickPositions(i) == 0 || yTickPositions(i) == y_max || yTickPositions(i) == -y_max % Show label for even indices
        yTickLabels{i} = num2str(yTickPositions(i));  % Set label as the YTick value
    else
        yTickLabels{i} = '';  % Leave odd-index labels blank
    end
end
% 
set(gca, 'XTick', x_ticks)
set(gca, 'YTick', yTickPositions)
set(gca, 'xGrid', 'on')
set(gca, 'yGrid', 'on')
% set(gca, 'YTickLabel', yTickLabels);
set(gca, 'YTickLabel', '');
set(gca, 'xTickLabel','')



subplot(4,4,[12 16]);  
pcolorjk(ts(time_start:time_end), q_133.frx(q_133.frx>=1), spectrogram(:,time_start:time_end));
caxis([-6 6]);

set(gca,'yscale','log');
yt=[2 4 10 25 70 200];
% set(gca, "yTick", yt,'YTickLabel',cellstr(num2str(round(yt)'))')
set(gca, "yTick",[])
ax = gca;               % Get current axes
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;

% set(gca, "YTickLabel",yTickLabels)

hold on; plot(xlim,[70 70],'w-', 'LineWidth',5);
hold on; plot(xlim,[150 150],'w-', 'LineWidth',5);
hold on; plot([0 0],ylim,'k-');
h = gcf; 
c = colorbar('Position', [0.93, 0.1, 0.02, 0.382]);
c.FontSize = 24; % Set larger font size for the color bar text








