load('/data/an/transformed/RR_h_referential_wSPKS/EC219/AN/EC219_AN_ch402_wSPKS.mat')
count = 0;
overall_raw = [];
overall_filtered = [];
overall_spec = [];
overall_spec_filtered = [];
slidewin=0.1;
% data_points = int32(513/2)*(0.06/0.5)-2;
data_points = int32(q.sfx/2)*(slidewin/0.5)-2;
disp(data_points)

root_dir = '/data/an/transformed/RR_h_referential_wSPKS/EC137/AN/';
channel = 'EC137_AN_ch272_wSPKS.mat';
for i=[257:260,269:273]
    result = load([root_dir 'EC137_AN_ch' num2str(i) '_wSPKS.mat']);
    q = [result.q];
    count = 0;
    while count <2000;
        block_of_interest = randi([1, length(q.Rdata)]);
        if length(q.Rdata(block_of_interest).Rspikes_ANYchannel) == 0
            continue;
        end
        candi_of_interest = randi([1, length(q.Rdata(block_of_interest).Rspikes_ANYchannel)]);
        
    
        count = count+1;
        disp(count)
    
        ts=getts(size(q.Rdata(block_of_interest).Rspects, 2), q.sfx)-.5;
         
        spectrogram=sq(q.Rdata(block_of_interest).Rspects(candi_of_interest,:,:))';
        spectrogram_filtered = spectrogram(49:60,:);
        raw_trace = sq(q.Rdata(block_of_interest).Rtraces(candi_of_interest,:));
        filtered_trace = 2*jkfilt(q.Rdata(block_of_interest).Rtraces(candi_of_interest,:), q.sfx,70,150);


        result_spec_filtered = [];
        for i=50:462
        % for i=data_points+1:512-data_points
            moving = make1d(spectrogram_filtered(:,i-data_points:i+data_points));
            reference = make1d(spectrogram_filtered(:, 256-data_points:256+data_points));
            result_spec_filtered = [result_spec_filtered; core(moving, reference)];
        end
        overall_spec_filtered= [overall_spec_filtered;result_spec_filtered(:)'];

        
        
        result_spec = [];
        for i=50:462
        % for i=data_points+1:512-data_points
            moving = make1d(spectrogram(:,i-data_points:i+data_points));
            reference = make1d(spectrogram(:, 256-data_points:256+data_points));
            result_spec = [result_spec; core(moving, reference)];
        end
        overall_spec= [overall_spec;result_spec(:)'];

  
        result_filter = [];
        % for i=data_points+1:512-data_points
        for i=50:462
            % shifted_A = circshift(filtered_trace, i);
            moving = filtered_trace(i-data_points:i+data_points);
            reference = filtered_trace(256-data_points:256+data_points);
            result_filter = [result_filter;(core(moving, reference))];
        end
        overall_filtered= [overall_filtered;result_filter(:)'];



        result_raw = [];
        % for i=data_points+1:512-data_points
        for i=50:462
            moving = raw_trace(i-data_points:i+data_points);
            reference = raw_trace(256-data_points:256+data_points);
            result_raw = [result_raw;(core(moving, reference))];
        end
        overall_raw= [overall_raw;result_raw(:)'];

    end
end

avg_raw = mean(overall_raw, 1); % Average across rows
avg_filtered = mean(overall_filtered, 1); % Average across rows
avg_spec = mean(overall_spec, 1); % Average across rows
avg_spec_filtered = mean(overall_spec_filtered, 1); % Average across rows
figure;
% 
% x = linspace( (50/513)*0.5, (450/513)*0.5 , length(50:450));
disp(ts)
x = ts(50:462);

disp(size(overall_raw))
disp(size(overall_spec))
disp(size(overall_filtered))

plot(x, avg_spec,'color', [0.1 0.5 0.1], LineWidth=2);; hold on
% ribbons(x,overall_spec, [0.1 0.5 0.1],[],'ci');
hold on

% plot(x, avg_spec_filtered, 'color',[0.1 0.9 0.1], LineWidth=2); hold on
% % ribbons(x,overall_spec_filtered,[0.1 0.9 0.1],[],'ci');
% hold on

plot(x, avg_raw,'color', [0.5 0.1 0.5], LineWidth=2); hold on
% ribbons(x,overall_raw, [0.9 0.1 0.9],[],'ci');
hold on

plot(x, avg_filtered, 'color', [0.9 0.1 0.9], LineWidth=2); hold on
% ribbons(x,overall_filtered,[0.5 0.1 0.5],[],'ci');
hold on
% disp(column_avg);
% xlim([data_points+1, 512-data_points])
xlim([ts(50) ts(462)]);
% legend('Raw\_trace\_Spectrogram', '','Raw\_trace\_Spectrogram, 95% CI', '','Filtered\_trace\_Spectrogram','','Filtered\_trace\_Spectrogram, 95% CI', 'Raw\_trace', 'Raw\_trace, 95% CI','Filtered\_trace','Filtered\_trace, 95% CI');
% legend('','Raw\_trace\_Spectrogram','', '', 'Filtered\_trace\_Spectrogram','','', 'Raw\_trace', '','','Filtered\_trace','');
% 
%legend

grid on
set(gcf,'color','w')