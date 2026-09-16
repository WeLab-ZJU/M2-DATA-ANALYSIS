bin_ar_20hz = reshape(R_ar,2,[]);
Area_Change_20hz = mean(bin_ar_20hz,1);
% Area_Change_20hz = (Area_Change_20hz-mean(Area_Change_20hz))/std(Area_Change_20hz);
% Area_Change_20hz = (Area_Change_20hz-min(Area_Change_20hz))/(max(Area_Change_20hz)-min(Area_Change_20hz));

r        = 7;
sigma    = 4;
Area_Change_20hz_filter = Gaussianfilter(r, sigma, Area_Change_20hz);
Area_Change_20hz_filter_temp = diff(Area_Change_20hz_filter);
Area_Change_20hz_filter_temp = abs(Gaussianfilter(r, sigma, Area_Change_20hz_filter_temp));

% im_20hz=Area_Change_20hz_filter_temp<0.020;
im_20hz=Area_Change_20hz_filter_temp<0.004;
im_20hz_smooth = zeros(size(im_20hz));

for i = 1:length(im_20hz)
    if i<=20
        % head
        status_tem = sum(im_20hz(1,1:(i+20)));
        if status_tem < length(im_20hz(1,1:(i+20)))/2
            im_20hz_smooth(i) = 0;
        else
            im_20hz_smooth(i) = 1;
        end
    else
        if i>(length(im_20hz)-20)
            % tail
            status_tem = sum(im_20hz(1,i:end));
            if status_tem < length(im_20hz(1,i:end))/2
                im_20hz_smooth(i) = 0;
            else
                im_20hz_smooth(i) = 1;
            end
        else
            % intermedian
            status_tem = sum(im_20hz(1,i-20:i+20));
            if status_tem < length(im_20hz(1,i-20:i+20))/2
                im_20hz_smooth(i) = 0;
            else
                im_20hz_smooth(i) = 1;
            end
        end
    end
end

Area_Change_20hz_table = table(Area_Change_20hz);
Area_Change_20hz_filter_table = table(Area_Change_20hz_filter);
im_20hz_smooth_table = table(im_20hz_smooth);
writetable(Area_Change_20hz_table, 'Area_Change_raw.csv');
writetable(Area_Change_20hz_filter_table, 'Area_Change.csv');
writetable(im_20hz_smooth_table, 'im.csv');

clc;clear all