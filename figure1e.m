load("trace_paper_corr.mat");
plot(x, avg_spec,'color', [0.1 0.5 0.1], LineWidth=2); hold on
%ribbons(x,overall_spec, [0.1 0.5 0.1],[],'ci');
hold on
% plot(x, avg_spec_filtered, 'color',[0.1 0.9 0.1], LineWidth=2); hold on
% %ribbons(x,overall_spec_filtered,[0.1 0.9 0.1],[],'ci');
% hold on
plot(x, avg_raw,'color', [0.9 0.1 0.9], LineWidth=2); hold on
%ribbons(x,overall_raw, [0.9 0.1 0.9],[],'ci');
hold on
plot(x, avg_filtered, 'color',[0.5 0.1 0.5], LineWidth=2); hold on
%ribbons(x,overall_filtered,[0.5 0.1 0.5],[],'ci');
hold on
% disp(column_avg);
% xlim([data_points+1, 512-data_points])
xlim([ts(50) ts(462)]);
% legend('Raw\_trace\_Spectrogram', '','Raw\_trace\_Spectrogram, 95% CI', '','Filtered\_trace\_Spectrogram','','Filtered\_trace\_Spectrogram, 95% CI', 'Raw\_trace', 'Raw\_trace, 95% CI','Filtered\_trace','Filtered\_trace, 95% CI');
% legend('','Raw\_trace\_Spectrogram','', '', 'Filtered\_trace\_Spectrogram','','', 'Raw\_trace', '','','Filtered\_trace','');
grid on
set(gcf,'color','w')
set(gca,'ylim', [-1 1])
