function get_best_param_v2(usecdi)
usecdi = true;
if usecdi
    pth = '../grid_search_withcdi';
else
    pth = '../grid_search_nocdi';
end
    
[datastrin, ep, b, d, lr, l1, l2, lstmi, densei, h, s]  = ...
    load_vals(pth, 'lstm_w');
% save(fullfile(pth, 'grid_search_summary_waves.mat'), 'datastrin', 'ep', ...
%     'b', 'd', 'lr', 'l1', 'l2', 'lstmi', 'densei', 'h', 's');


[datastrin, ep, b, d, lr, l1, l2, lstmi, densei, h, s]  = ...
    load_vals(pth, 'lstm_old');
save(fullfile(pth, 'grid_search_summary_old.mat'), 'datastrin', 'ep', ...
    'b', 'd', 'lr', 'l1', 'l2', 'lstmi', 'densei', 'h', 's');

[datastrin, ep, b, d, lr, l1, l2, lstmi, densei, h, s]  = ...
    load_vals(pth, 'lstm_fe');
save(fullfile(pth, 'grid_search_summary_fe.mat'), 'datastrin', 'ep', ...
    'b', 'd', 'lr', 'l1', 'l2', 'lstmi', 'densei', 'h', 's');
[datastrin, ep, b, d, lr, l1, l2, lstmi, densei, h, s]  = ...
    load_vals(pth, 'lstm_m');
save(fullfile(pth, 'grid_search_summary_m.mat'), 'datastrin', 'ep', ...
    'b', 'd', 'lr', 'l1', 'l2', 'lstmi', 'densei', 'h', 's');
[datastrin, ep, b, d, lr, l1, l2, lstmi, densei, h, s]  = ...
    load_vals(pth, 'lstm_young');
save(fullfile(pth, 'grid_search_summary_young.mat'), 'datastrin', 'ep', ...
    'b', 'd', 'lr', 'l1', 'l2', 'lstmi', 'densei', 'h', 's');

end
function [datastrin, ep, b, d, lr, l1, l2, lstmi, densei, h, s]  = load_vals(pth, str1)
figure;
for win = 1:3
    
    r2vals=[];ss=[];hh=[];dd=[];ee=[];lrr=[];ll1=[];ll2=[];
    files = dir(fullfile(pth, [str1 num2str(win)  '_*.mat']));
    for fi = 1:length(files)
        currres = load(fullfile(files(fi).folder, files(fi).name));
        r2vals(fi) = mean(max(0,currres.r2_all_dev));
        hh(fi) = currres.conf_all.h;
        ss(fi) = currres.conf_all.s;
        ee(fi) = currres.conf_all.e;
        dd(fi) = currres.conf_all.d;
        lrr(fi) = currres.conf_all.lr;
        ll1(fi) = currres.conf_all.l1;
        ll2(fi) = currres.conf_all.l2;
        r2vals_all(:, fi) = max(0,currres.r2_all_dev);
    end
    if isempty(r2vals)
        continue;
    end
%     subplot(6,3,win);scatter(ss, r2vals, '.');ylim([0.5 0.9])
%     subplot(6,3,3+win);scatter(hh, r2vals, '.');ylim([0.5 0.9])
%     subplot(6,3,6+win);scatter(dd, r2vals, '.');ylim([0.5 0.9])
%     subplot(6,3,9+win);scatter(ll1, r2vals, '.');ylim([0.5 0.9])
%     subplot(6,3,12+win);scatter(ll2, r2vals, '.');ylim([0.5 0.9])
%     subplot(6,3,15+win);scatter(lrr, r2vals, '.');ylim([0.5 0.9])
    [maxv, maxi] = max(r2vals);
    currres = load(fullfile(files(maxi).folder, files(maxi).name));
    
    datastrin{win} = [str1(6:end) num2str(win)];
    
    ep(win) = currres.conf_all.e;
    b(win) = currres.conf_all.b;
    d(win) = currres.conf_all.d;
    lr(win) = currres.conf_all.lr;
    l1(win) = currres.conf_all.l1;
    l2(win) = currres.conf_all.l2;
    lstmi(win) = currres.conf_all.lstmi;
    densei(win) = currres.conf_all.densei;
    s(win) = currres.conf_all.s;
    h(win) = currres.conf_all.h;
   
    
    r2vals(fi) = mean(currres.r2_all_dev);
    
    indic = hh == h(win) & ee == ep(win) & dd == d(win) & ...
        lrr == lr(win) & ll1 == l1(win) & ll2 == l2(win); 
    x = ss(indic);
    y = r2vals(indic);
    [~,ic] = sort(x);
    subplot(1,3,win);
    Y = r2vals_all(:, indic);
    X = repmat(x, size(Y, 1), 1);
    plot(X(:, ic)', Y(:, ic)', 'x-');
    hold all;bh = errorbar(x(ic), mean(Y(:, ic)), std(Y(:, ic))/sqrt(4));
    bh.Color = 'k';
    bh.LineWidth = 2;
%     plot(x(ic), y(ic));%ylim([0.6 0.8])
    title(datastrin{win});
    if win == 2, xlabel('Seq. Len.');end
    if win == 1, ylabel('Accuracy');end
end
end
function bla
figure;y=1;

for win = 1:3
    for wout = 1:3
        r2vals=[];ss=[];
        files = dir(fullfile(pth, ['lstm_fe' num2str(win) '_m' num2str(wout) '_*.mat']));
        for fi = 1:length(files)
            currres = load(fullfile(files(fi).folder, files(fi).name));
            r2vals(fi) = mean(currres.r2_all_dev);
            st = strfind(files(fi).name, 'sequenceList')+12;
            en = strfind(files(fi).name, '_hidden_size')-1;
            ss(fi) = str2double(files(fi).name(st:en));
        end
        if isempty(r2vals)
            continue;
        end
        subplot(3,3,y);scatter(ss,r2vals, '.');title(['f' num2str(win) ' m' num2str(wout)]);y=y+1;
        
        [maxv, maxi] = max(r2vals);
        currres = load(fullfile(files(maxi).folder, files(maxi).name));
        
        datastrin{l} = ['fe' num2str(win)];
        datastrout{l} = ['m' num2str(wout)];
        
        ep(l) = currres.conf_all.e;
        b(l) = currres.conf_all.b;
        d(l) = currres.conf_all.d;
        lr(l) = currres.conf_all.lr;
        l1(l) = currres.conf_all.l1;
        l2(l) = currres.conf_all.l2;
        lstmi(l) = currres.conf_all.lstmi;
        densei(l) = currres.conf_all.densei;
        s(l) = currres.conf_all.s;
        h(l) = currres.conf_all.h;
        l=l+1;
    end
end

save('grid_search_summary.mat');
end