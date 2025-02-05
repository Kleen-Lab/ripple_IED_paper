function [V,F,rgbmat,labels] = hp_spike_mni()

% getbrain('MNI',1,0,'b');
[~,lhippmesh]=highlightdeep('mni','hp',[0.4 0.4 0.4;1 1 1],[0.15 0],0,0,'l');
    lhippmesh=lhippmesh.cortex.vert;
[~,rhippmesh]=highlightdeep('mni','hp',[0.4 0.4 0.4;1 1 1],[0.15 0],0,0,'r');
    rhippmesh=rhippmesh.cortex.vert;
d = dictionary();
d('EC133') = "#ff7f0e";
d('EC143') = "#e377c2";
d('EC157') = "#9467bd";
d('EC162') = "#c49c94";
% d('EC168') = "c";
d('EC175') = "#98df8a";

% d('EC181') = "k";
d('EC183') = "#c7c7c7";
d('EC186') = "#1f77b4";
d('EC187') = "#bcbd22";
d('EC191') = "#ffbb78";
d('EC196') = "#2ca02c";
d('EC219') = "#8c564b";
d('EC220') = "#d62728";
d('EC221') = "#ff9896";
d('EC222') = "#aec7e8";

d('EC130') = "#7f7f7f";
d('EC135') = "#c5b0d5";
d('EC137') = "#f7b6d2";


elecs_actual_d = dictionary();


total_sum = 0;
for p = ["EC130","EC133","EC135","EC137","EC143","EC157","EC162","EC175","EC183","EC186","EC187","EC191","EC196","EC219","EC220","EC221","EC222"]
    patient = convertStringsToChars(p);
 disp(patient)
%  disp(cell2mat(d(patient)))
em=getelecs(patient,3);

[elecmatrix,eleclabels,anatomy]=getelecs(patient,2);
% hp_region_patient = getelecs_region(patient,'hp',2);
% hp_region_patient = getregionelecs_verified2(patient,'hp');
if patient == 'EC133'
    hp_region_patient = [282, 291, 292, 293];
elseif patient == 'EC143'
    hp_region_patient = [339];
elseif patient == 'EC157'
    hp_region_patient = [321, 332, 333, 334];
elseif patient == 'EC162'
    hp_region_patient = [85, 86];
elseif patient == 'EC175'
    hp_region_patient = [301, 302];
elseif patient == 'EC183' 
    hp_region_patient = [290];
elseif patient == 'EC186' 
    hp_region_patient = [299, 300, 301];
elseif patient == 'EC187'
    hp_region_patient = [309, 310];
elseif patient == 'EC191'
    hp_region_patient = [2, 3, 4];
elseif patient =='EC196'
    hp_region_patient = [300, 301, 303];
elseif patient ==('EC219')
    hp_region_patient = [399, 400, 402, 403];
elseif patient == 'EC220' 
    hp_region_patient = [53, 54, 76, 77];
elseif patient == 'EC221' 
    hp_region_patient = [376, 377];
elseif patient == 'EC222' 
    hp_region_patient = [267, 268, 269, 270];
elseif patient ==('EC130') 
%     hp_region_patient = [1, 3];
     hp_region_patient = [1, 13];
elseif patient ==('EC135') 
    hp_region_patient = [303];
elseif patient ==('EC137') 
    hp_region_patient = [257, 258, 259, 269];
end


a = (size(hp_region_patient));
disp(num2str(a(2)))
total_sum = total_sum + a(2);
annotations = anatomy(:,4);

% snap electrodes to the closest vertex of the left or right hippocampal mesh

for e=1:size(em,1);
    if em(e,1)<0;
        em(e,:)=snapelectrodetomesh(em(e,:),lhippmesh);
    else
        em(e,:)=snapelectrodetomesh(em(e,:),rhippmesh);
    end
end


labels={};
rgbmat=[];
% for c = 1:nch
for c = hp_region_patient
    
    disp(c)
    A = (annotations(c));
    anno = A{1};
%     disp(anno)
    plot3(em(c,1),   em(c,2) , em(c,3), '.','Color',cell2mat(d(patient)),'MarkerSize', 30);
end
end



disp(['total  ' num2str(total_sum)])
litebrain('i',0)
