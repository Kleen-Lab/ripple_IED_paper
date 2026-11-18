
clearvars;

%% ==================== Configuration ====================

subject = 'sub-13';

% Original data directory
baseDir = '/Users/dazhang/Downloads/ieeg_ieds_bids';

% Root directory for new results
outputRoot = '/Users/dazhang/Downloads/IED_spectrograms_no_positive_overlap';

% Patient-specific output directory
outputDir = fullfile(outputRoot, subject);

% Patient-specific figure directory
figDir = fullfile(outputDir, 'figures');

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

% Save one comparison figure for each candidate
saveFigures = true;

% Sampling rates
sfx = 1000;
sfx512 = 512;

% Spectrogram parameters
numberOfFreqs = 61;
frxrange = [1 200];
voz = 8;
z = 1;  % z-score

% Window parameters
negShift = 3;      % Start searching 5 seconds before IED
scanStep = 1;      % If overlap, try 6s, 7s, 8s... before IED
halfWindow = 51;   % Samples on each side
nTime = 2*halfWindow + 1;

% Relative time axis, in seconds
timeAxis = (-halfWindow:halfWindow) / sfx512;

%% ==================== Load EDF Data ====================

edfFile = fullfile(baseDir, subject, 'ieeg', ...
    sprintf('%s_task-sleep_ieeg.edf', subject));

[data, annotations] = edfread(edfFile);

% Follow original code convention:
% annotations = channels x samples
d = annotations;

assert(isnumeric(d) && ismatrix(d), ...
    'Expected numeric channels x samples from edfread.');

%% ==================== Load Channel Information ====================

channelsFile = fullfile(baseDir, subject, 'ieeg', ...
    sprintf('%s_task-sleep_channels.tsv', subject));

opts = detectImportOptions(channelsFile, 'FileType', 'text');
opts.VariableNamingRule = 'preserve';

M = readtable(channelsFile, opts);

nameColumn = find(strcmpi(M.Properties.VariableNames, 'name'), 1);
channelNames = strtrim(string(M{:,nameColumn}));

channelMap = table(channelNames, ...
    (1:numel(channelNames))', ...
    'VariableNames', {'channel_name', 'channel_number'});

channelDict = dictionary( ...
    channelMap.channel_name, channelMap.channel_number);

assert(size(d,1) >= numel(channelNames), ...
    'EDF data has fewer channels than channels.tsv.');

%% ==================== Load Events ====================

eventsFile = fullfile(baseDir, subject, 'ieeg', ...
    sprintf('%s_task-sleep_events.tsv', subject));

opts = detectImportOptions(eventsFile, ...
    'FileType', 'text', 'Delimiter', '\t');

opts.VariableNamingRule = 'preserve';
opts = setvartype(opts, 'trial_type', 'string');

T = readtable(eventsFile, opts);

% Preserve original conversion from 1000 Hz to 512 Hz
eventCenters_512 = round(T.sample * sfx512 / sfx);

% Positive windows
ONOFFsets_512 = [ ...
    eventCenters_512 - halfWindow, ...
    eventCenters_512 + halfWindow];

% All positive windows (for overlap checks across ALL event times).
% These intervals are inclusive: [start_sample, end_sample].
% We do not precompute negative windows: each negative is found by scanning.
allPositiveWindows = ONOFFsets_512;


%% ==================== Load Event Channels ====================

interpretationFile = fullfile(baseDir, 'derivatives', ...
    sprintf('%s_task-sleep_events_interpretation.tsv', subject));

opts = detectImportOptions(interpretationFile, 'FileType', 'text');
opts.VariableNamingRule = 'preserve';

N = readtable(interpretationFile, opts);

event_chans = cellfun( ...
    @(x) strsplit(strtrim(x)), ...
    cellstr(string(N.chans)), ...
    'UniformOutput', false);

assert(height(T) == height(N), ...
    'Event and interpretation tables have different row counts.');

%% ==================== Initialize Output ====================

% Spectrograms
ls_spec = zeros(0, numberOfFreqs*nTime);
ls_spec_neg = zeros(0, numberOfFreqs*nTime);

% EEG traces
ls_trace = zeros(0, nTime);
ls_trace_neg = zeros(0, nTime);

% Metadata
event_ids = [];
channel_ids = [];
sample_channel_names = strings(0,1);

% One selected negative window per retained event/channel pair
selected_neg_windows_512 = zeros(0,2);
selected_neg_shifts_sec = zeros(0,1);


%% ==================== Generate Spectrograms & Traces ====================

% Unique channels associated with IED events
all_event_chans = string([event_chans{:}]);
all_event_chans = unique(all_event_chans);
all_event_chans = all_event_chans(strlength(all_event_chans) > 0);

for c = 1:numel(all_event_chans)

    channel_name = all_event_chans(c);

    if ~isKey(channelDict, channel_name)
        warning('Channel %s not found. Skipping.', channel_name);
        continue;
    end

    channel_number = channelDict(channel_name);

    fprintf('\nProcessing channel %d/%d: %s\n', ...
        c, numel(all_event_chans), channel_name);

    %% -------- Preprocessing --------

    channel_data = double(d(channel_number,:));

    % Anti-aliasing filter
    channel_data = jkfilt( ...
        channel_data, sfx, 0.1, floor(sfx512/2)-1);

    % Resample from 1000 Hz to 512 Hz
    channel_data_512 = resample( ...
        channel_data, sfx512, sfx);

    % Convert to row vector
    channel_data_512 = channel_data_512(:)';

    %% -------- Hilbert Spectrogram --------

    [h_z,~,frx,~] = zAAPHZ( ...
        channel_data_512, ...
        sfx512, frxrange, z, 1/voz);

    % Expected output: time x frequency
    assert(numel(h_z) == ...
        numel(channel_data_512)*numberOfFreqs, ...
        'Unexpected zAAPHZ output dimensions.');

    h_z = reshape(h_z, [], numberOfFreqs);

    %% -------- Extract Events for Current Channel --------

    for i = 1:height(T)

        current_chans = string(event_chans{i});

        if ~any(current_chans == channel_name)
            continue;
        end

        % Positive indices
        pos_idx = ONOFFsets_512(i,1): ...
                  ONOFFsets_512(i,2);

        % Check positive window bounds first
        if pos_idx(1) < 1 || pos_idx(end) > size(h_z,1)
            fprintf('Skipping event %d: positive window out of bounds.\n', i);
            continue;
        end

        %% -------- Find non-overlapping negative window --------
        % Start at -5 seconds. If it overlaps ANY positive window,
        % scan backward in 1-second increments (-6s, -7s, ...).
        % Only intervals fully inside the recording are allowed.
        foundNegative = false;
        negCenter = NaN;
        negStart = NaN;
        negEnd = NaN;

        stepSamples = round(scanStep * sfx512);
        assert(stepSamples >= 1, 'scanStep must be positive.');
        initialCenter = eventCenters_512(i) - round(negShift*sfx512);

        for candidateCenter = initialCenter:-stepSamples:(halfWindow+1)
            candidateStart = candidateCenter - halfWindow;
            candidateEnd   = candidateCenter + halfWindow;

            % A candidate [a,b] overlaps a positive [c,d] iff
            % a <= d AND b >= c (including touching endpoints).
            overlapsPositive = any( ...
                candidateStart <= allPositiveWindows(:,2) & ...
                candidateEnd >= allPositiveWindows(:,1));

            if ~overlapsPositive && candidateEnd <= size(h_z,1)
                foundNegative = true;
                negCenter = candidateCenter;
                negStart = candidateStart;
                negEnd = candidateEnd;
                break;
            end
        end

        if ~foundNegative
            fprintf('Skipping event %d, channel %s: no valid prior negative.\n', ...
                i, char(channel_name));
            continue;
        end

        neg_idx = negStart:negEnd;
        actualNegShiftSec = (eventCenters_512(i) - negCenter)/sfx512;

        % Verification: no selected negative window overlaps any positive
        assert(~any(negStart <= allPositiveWindows(:,2) & ...
                    negEnd >= allPositiveWindows(:,1)), ...
               'Negative sample overlaps an annotated positive window.');

        %% -------- Extract Spectrograms --------

        % Each spectrogram: 61 x 103
        spec_pos = h_z(pos_idx,:)';
        spec_neg = h_z(neg_idx,:)';

        assert(isequal(size(spec_pos), ...
            [numberOfFreqs,nTime]), ...
            'Unexpected positive spectrogram size.');

        assert(isequal(size(spec_neg), ...
            [numberOfFreqs,nTime]), ...
            'Unexpected negative spectrogram size.');

        %% -------- Extract EEG Traces --------

        % Each trace: 1 x 103
        trace_pos = channel_data_512(pos_idx);
        trace_neg = channel_data_512(neg_idx);

        %% -------- Store Data --------

        % Spectrograms: N x 6283
        ls_spec(end+1,:) = make1d(spec_pos);
        ls_spec_neg(end+1,:) = make1d(spec_neg);

        % EEG traces: N x 103
        ls_trace(end+1,:) = trace_pos;
        ls_trace_neg(end+1,:) = trace_neg;

        % Metadata
        event_ids(end+1,1) = i;
        channel_ids(end+1,1) = channel_number;
        sample_channel_names(end+1,1) = channel_name;
        selected_neg_windows_512(end+1,:) = [negStart negEnd];
        selected_neg_shifts_sec(end+1,1) = actualNegShiftSec;

        %% -------- Save Comparison Figure --------

        if saveFigures

            sample_idx = size(ls_spec,1);

            fig = figure( ...
                'Visible','off', ...
                'Color','w', ...
                'Position',[150 150 1100 750]);

            % Positive trace
            ax1 = subplot(2,2,1);
            plot(timeAxis, trace_pos, ...
                'r', 'LineWidth',1.2);
            xlabel('Time (s)');
            ylabel('iEEG amplitude');
            title('Positive IED Trace');
            grid on;

            % Negative trace
            ax2 = subplot(2,2,2);
            plot(timeAxis, trace_neg, ...
                'b', 'LineWidth',1.2);
            xlabel('Time (s)');
            ylabel('iEEG amplitude');
            title(sprintf('Negative Trace (-%.1fs)', actualNegShiftSec));
            grid on;

            % Match trace y-axis limits
            allY = [trace_pos trace_neg];

            if all(isfinite(allY))
                ymin = min(allY);
                ymax = max(allY);

                if ymin < ymax
                    ylim(ax1,[ymin ymax]);
                    ylim(ax2,[ymin ymax]);
                end
            end

            % Positive spectrogram
            subplot(2,2,3);
            pcolor(timeAxis, frx, spec_pos);
            shading flat;
            set(gca,'YScale','log');
            caxis([-2 2]);
            colorbar;
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            title('Positive IED Spectrogram');

            % Negative spectrogram
            subplot(2,2,4);
            pcolor(timeAxis, frx, spec_neg);
            shading flat;
            set(gca,'YScale','log');
            caxis([-2 2]);
            colorbar;
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            title(sprintf('Negative Spectrogram (-%.1fs)', actualNegShiftSec));

            sgtitle(sprintf( ...
                '%s | Event %d | Channel %s | Candidate %d', ...
                subject, i, char(channel_name), sample_idx));

            % Save PNG in patient-specific figures folder
            figFile = fullfile(figDir, ...
                sprintf( ...
                'candidate_%04d_event_%04d_channel_%s.png', ...
                sample_idx, i, char(channel_name)));

            exportgraphics(fig, figFile, 'Resolution',200);

            close(fig);

        end

    end

    % Free memory
    clear h_z channel_data channel_data_512;

end

%% ==================== Verify Results ====================

fprintf('\n========== Results ==========\n');

fprintf('Positive spectrograms: %d x %d\n', ...
    size(ls_spec));

fprintf('Negative spectrograms: %d x %d\n', ...
    size(ls_spec_neg));

fprintf('Positive traces: %d x %d\n', ...
    size(ls_trace));

fprintf('Negative traces: %d x %d\n', ...
    size(ls_trace_neg));

nSamples = size(ls_spec,1);

assert(size(ls_spec_neg,1) == nSamples && ...
       size(ls_trace,1) == nSamples && ...
       size(ls_trace_neg,1) == nSamples, ...
       'Positive and negative sample counts do not match.');

%% ==================== Save Spectrograms ====================

outputFilePos = fullfile(outputDir, ...
    sprintf('%s_ls_spec.mat', subject));

save(outputFilePos, 'ls_spec', '-v7.3');

outputFileNeg = fullfile(outputDir, ...
    sprintf('%s_ls_spec_neg.mat', subject));

save(outputFileNeg, 'ls_spec_neg', '-v7.3');

%% ==================== Save EEG Traces ====================

outputTracePos = fullfile(outputDir, ...
    sprintf('%s_ls_trace.mat', subject));

save(outputTracePos, ...
    'ls_trace', 'timeAxis', '-v7.3');

outputTraceNeg = fullfile(outputDir, ...
    sprintf('%s_ls_trace_neg.mat', subject));

save(outputTraceNeg, ...
    'ls_trace_neg', 'timeAxis', '-v7.3');

%% ==================== Save Metadata ====================

metadataFile = fullfile(outputDir, ...
    sprintf('%s_spec_metadata.mat', subject));

save(metadataFile, ...
    'event_ids', ...
    'channel_ids', ...
    'sample_channel_names', ...
    'ONOFFsets_512', ...
    'selected_neg_windows_512', ...
    'selected_neg_shifts_sec', ...
    'allPositiveWindows', ...
    'scanStep', ...
    'negShift', ...
    'frx', ...
    'timeAxis', ...
    'sfx', ...
    'sfx512', ...
    'numberOfFreqs', ...
    'halfWindow', ...
    '-v7.3');

%% ==================== Finished ====================

fprintf('\n=======================================\n');
fprintf('Subject: %s\n', subject);
fprintf('Total matched candidates: %d\n', nSamples);
if nSamples > 0
    fprintf('Actual negative offsets: %.1f to %.1f seconds before IED\n', ...
        min(selected_neg_shifts_sec), max(selected_neg_shifts_sec));
end
fprintf('Output directory:\n%s\n', outputDir);
fprintf('Figure directory:\n%s\n', figDir);
fprintf('=======================================\n');
