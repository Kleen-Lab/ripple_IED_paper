
function [raw_trace, filtered_trace, spectrogram, ts] = plot_trace_filter_spec(q, block_of_interest, candi_of_interest);

disp(block_of_interest)
disp(candi_of_interest)

ts=getts(size(q.Rdata(block_of_interest).Rspects, 2), q.sfx)-.5;
disp(length(ts));
spectrogram=sq(q.Rdata(block_of_interest).Rspects(candi_of_interest,:,:))';
raw_trace = sq(q.Rdata(block_of_interest).Rtraces(candi_of_interest,:));
filtered_trace = jkfilt(q.Rdata(block_of_interest).Rtraces(candi_of_interest,:), q.sfx,70,150);
