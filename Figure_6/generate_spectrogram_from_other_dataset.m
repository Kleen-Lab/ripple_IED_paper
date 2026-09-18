% "ReferencesAndLinks": ["https://doi.org/10.1038/s41597-024-04187-y", "https://doi.org/10.6084/m9.figshare.26131978"],
% "DatasetDOI": "10.6084/m9.figshare.26131978"
% Falach, R., Geva-Sagiv, M., Eliashiv, D. et al. Annotated interictal discharges in intracranial EEG sleep data and related machine learning detection scheme. Sci Data 11, 1354 (2024). https://doi.org/10.1038/s41597-024-04187-y
% The data is saved in Box folder ~/Box/KLEENLAB/-LAB MEMBERS/Da/ieeg_ieds_bids

subject = 'sub-25';
baseDir = '/Users/dazhang/Downloads/ieeg_ieds_bids';
sfx512=512;
sfx=1000;
edfFile = fullfile(baseDir, subject, 'ieeg', sprintf('%s_task-sleep_ieeg.edf', subject));


[data, annotations] = edfread(edfFile);
filename = fullfile(baseDir, subject,'ieeg',sprintf('%s_task-sleep_channels.tsv', subject));
opts = detectImportOptions(filename, 'FileType', 'text');
opts.VariableNamingRule = 'preserve';
M = readtable(filename, opts);
nameColumn = find(strcmpi(M.Properties.VariableNames, 'name'), 1);
channelNames = strtrim(string(M{:, nameColumn}));
channelMap = table(channelNames, (1:numel(channelNames))','VariableNames', {'channel_name', 'channel_number'});
channelDict = dictionary(channelMap.channel_name, channelMap.channel_number);
% disp(channelMap)
% channelDict = containers.Map(cellstr(channelNames),num2cell(1:numel(channelNames)));


eventsFile = fullfile(baseDir, subject, 'ieeg', sprintf('%s_task-sleep_events.tsv', subject));
opts = detectImportOptions(eventsFile, 'FileType', 'text','Delimiter', '\t');
opts.VariableNamingRule = 'preserve';
opts = setvartype(opts, 'trial_type', 'string');
T = readtable(eventsFile, opts);
% events = [T.sample - 100, T.sample + 100];
% ONOFFsets_512=round(events(:,1:2)*(sfx512/sfx));
eventCenters_512 = round(T.sample * sfx512 / sfx);
ONOFFsets_512 = [eventCenters_512 - 51, eventCenters_512 + 51];


interpretationFile = fullfile(baseDir, 'derivatives', sprintf('%s_task-sleep_events_interpretation.tsv', subject));
opts = detectImportOptions(interpretationFile, 'FileType', 'text');
opts.VariableNamingRule = 'preserve';
N = readtable(interpretationFile, opts);
event_chans = cellfun(@(x) strsplit(strtrim(x)), N.chans, 'UniformOutput', false);
disp(T)

d512 = [];
d = annotations;

numberOfFreqs=61;
frxrange=[1 200];
voz=8;
z=1; % yes z-score
ls_spec = [];


for i=1:height(T)
%     s = events(i, 1);
%     e = events(i, 2);
    current_chans = event_chans{i};
    for j = 1:numel(current_chans)
        channel_name = current_chans{j};
        channel_number = channelDict(channel_name);
        fprintf('  Channel: %s %d \n', channel_name, channel_number);
%         this_channel_raw_data = annotations(channel_number,:);
%         d(channel_number,:)=jkfilt(d(channel_number,:), sfx, .1, floor(sfx512/2)-1); % antialisasing filter

        channel_data = double(annotations(channel_number, :));
        channel_data = jkfilt(channel_data, sfx, 0.1, floor(sfx512 / 2) - 1);
%         d512(channel_number,:)=d(channel_number, 1:sfx/sfx512:end); %downsampled to 512 Hz
        d512(channel_number, :) = resample(channel_data, sfx512, sfx);
        if channel_number==1;
            h_z=nan(length(d512(channel_number,:)),size(d,1),numberOfFreqs);
        end
        [h_z(:,channel_number,:),~,frx,~]=zAAPHZ(d512(channel_number,:), sfx512, frxrange, z, 1/voz); %hilbert analytic amplitude, z-scored
        spectrogram = squeeze(h_z(ONOFFsets_512(i,1):ONOFFsets_512(i,2),channel_number,:))';
        disp(size(spectrogram))
        ls_spec = [ls_spec; make1d(spectrogram)];
        figure; pcolor(ONOFFsets_512(i,1):ONOFFsets_512(i,2), frx, spectrogram)
        shading flat
        set(gca,'YScale','log')
%         edit EEGHILITE
        caxis([[-2 2]])
    end
end

disp(size(ls_spec))

outputDir = fullfile(baseDir, 'spectrograms');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
outputFile = fullfile(outputDir, sprintf('%s_ls_spec.mat', subject));
save(outputFile, 'ls_spec', '-v7.3');
fprintf('Saved %s: %d events × %d features\n', outputFile, size(ls_spec, 1), size(ls_spec, 2));




