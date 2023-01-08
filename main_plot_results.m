clear;
for n=1
% load('grid_search_summary_waves');
% for win = 1:3
%     wout = win;
%     seq_w(win) = s(find(strcmp(datastrin, ['w' num2str(win)]) & strcmp(datastrout, ['w' num2str(wout)])));
%     
% end
% load('grid_search_summary_m');
% 
% for win = 1:3
%     wout = win;
%     seq_m_m(win) = s(find(strcmp(datastrin, ['m' num2str(win)]) & strcmp(datastrout, ['fe' num2str(wout)])));
% end
% load('grid_search_summary_fe');
% 
% for win = 1:3
%     wout = win;
%     seq_f_f(win) = s(find(strcmp(datastrin, ['fe' num2str(win)]) & strcmp(datastrout, ['m' num2str(wout)])));
% end
% 
% load('grid_search_summary_old');
% 
% for win = 1:3
%     wout = win;
%     seq_o_o(win) = s(find(strcmp(datastrin, ['old' num2str(win)]) & strcmp(datastrout, ['young' num2str(wout)])));
% end
% load('grid_search_summary_young');
% 
% 
% for win = 1:3
%     wout = win;
%     seq_y_y(win) = s(find(strcmp(datastrin, ['young' num2str(win)]) & strcmp(datastrout, ['old' num2str(wout)])));
% end
end
withCDIsum = true;
if withCDIsum
feat_str = {'CDI_sum_mean', 'pos_inter_mom_sum_mean', 'pos_inter_mom_sum_dev', 'pos_inter_dad_sum_mean', 'pos_inter_dad_sum_dev', 'pos_inter_sibling_sum_mean', 'pos_inter_sibling_sum_dev', 'pos_inter_friend_part_sum_mean', 'pos_inter_friend_part_sum_dev', 'neg_inter_mom_sum_mean', 'neg_inter_mom_sum_dev', 'neg_inter_dad_sum_mean', 'neg_inter_dad_sum_dev', 'neg_inter_sibling_sum_mean', 'neg_inter_sibling_sum_dev', 'neg_inter_friend_part_sum_mean', 'neg_inter_friend_part_sum_dev'};
pth = '../results_w3_many2one_with_dev_norm_with_cdioutput';
else
    feat_str = { 'pos_inter_mom_sum_mean', 'pos_inter_mom_sum_dev', 'pos_inter_dad_sum_mean', 'pos_inter_dad_sum_dev', 'pos_inter_sibling_sum_mean', 'pos_inter_sibling_sum_dev', 'pos_inter_friend_part_sum_mean', 'pos_inter_friend_part_sum_dev', 'neg_inter_mom_sum_mean', 'neg_inter_mom_sum_dev', 'neg_inter_dad_sum_mean', 'neg_inter_dad_sum_dev', 'neg_inter_sibling_sum_mean', 'neg_inter_sibling_sum_dev', 'neg_inter_friend_part_sum_mean', 'neg_inter_friend_part_sum_dev'};
pth = '../results_w3_many2one_with_dev_norm_no_cdioutput';
end
str_leg=[];
feat_imp_w = nan(10, length(feat_str), 3);
feat_imp_m = nan(10, length(feat_str), 3);
feat_imp_f = nan(10, length(feat_str), 3);
feat_imp_y = nan(10, length(feat_str), 3);
feat_imp_o = nan(10, length(feat_str), 3);

for win = 1:3
    for wout = 1:3
        filename = ['lstm_w' num2str(win) '_w' num2str(wout) '_seq010.mat'];
        if ~isfile(fullfile(pth, filename))
            continue;
        end
        curr = load(fullfile(pth, filename));
        if win == wout
            waves_train(:, win) = max(0,curr.train_r2);
            
        end
        if length(curr.feature_weights)~=1
                feat_imp_w(:, :, win, wout) = squeeze((curr.feature_weights));
            else
                feat_imp_w(:, :, win, wout) = nan;
            end
        waves_test(:, win, wout) = max(0,curr.test_r2);
        
        str_leg{end+1} = ['w' num2str(win) '->w' num2str(wout)];
    end
end
for win = 1:3
    for wout = 1:3
        filename = ['lstm_m' num2str(win) '_fe' num2str(wout) '_seq010.mat'];
        if ~isfile(fullfile(pth, filename))
            continue;
        end
        curr = load(fullfile(pth, filename));
        if win == wout
            m_m_train(:, win) = max(0,curr.train_r2);
            m_m_test(:, win, wout) = max(0,curr.test_r2);
            
        end
        if length(curr.feature_weights)~=1
                feat_imp_m(:, :, win, wout) = squeeze(abs(curr.feature_weights));
            else
                feat_imp_m(:, :, win, wout) = nan;
            end
        m_f_test(:, win, wout) = max(0,curr.opp_r2);
        
    end
end

for win = 1:3
    for wout = 1:3
        filename = ['lstm_fe' num2str(win) '_m' num2str(wout) '_seq010.mat'];
        if ~isfile(fullfile(pth, filename))
            continue;
        end
        curr = load(fullfile(pth, filename));
        if win == wout
            f_f_train(:, win) = max(0,curr.train_r2);
            f_f_test(:, win, wout) = max(0,curr.test_r2);
            
        end
        if length(curr.feature_weights)~=1
                feat_imp_f(:, :, win, wout) = squeeze(abs(curr.feature_weights));
            else
                feat_imp_f(:, :, win, wout) = nan;
            end
        f_m_test(:, win, wout) = max(0,curr.opp_r2);
        
    end
end

for win = 1:3
    for wout = 1:3
        filename = ['lstm_young' num2str(win) '_old' num2str(wout) '_seq010.mat'];
        if ~isfile(fullfile(pth, filename))
            continue;
        end
        curr = load(fullfile(pth, filename));
        if win == wout
            y_y_train(:, win) = max(0,curr.train_r2);
            y_y_test(:, win, wout) = max(0,curr.test_r2);
            
        end
        if length(curr.feature_weights)~=1
                feat_imp_y(:, :, win, wout) = squeeze(abs(curr.feature_weights));
            else
                feat_imp_y(:, :, win, wout) = nan;
            end
        y_o_test(:, win, wout) = max(0,curr.opp_r2);
        
    end
end

for win = 1:3
    for wout = 1:3
        filename = ['lstm_old' num2str(win) '_young' num2str(wout) '_seq010.mat'];
        if ~isfile(fullfile(pth, filename))
            continue;
        end
        curr = load(fullfile(pth, filename));
        if win == wout
            o_o_train(:, win) = max(0,curr.train_r2);
            o_o_test(:, win, wout) = max(0,curr.test_r2);
            
        end
        if length(curr.feature_weights)~=1
                feat_imp_o(:, :, win, wout) = squeeze(abs(curr.feature_weights));
            else
                feat_imp_o(:, :, win, wout) = nan;
            end
        o_y_test(:, win, wout) = max(0,curr.opp_r2);
        
    end
end


M = mean(waves_train);
S = std(waves_train)/2;
M(2,:) = diag(squeeze(mean(waves_test)));
S(2,:) = diag(squeeze(std(waves_test)))/2;
figure;subplot(1,2,1);barwitherr(S',M');legend('Train','Test');ylabel('R2');xlabel('Waves');
M = mean(waves_test);
S = std(waves_test)/2;
subplot(1,2,2);barwitherr(S(:), M(:));setxlabels(str_leg);



M = diag(squeeze(mean(waves_test)));
S = diag(squeeze(std(waves_test)))/2;
M(:, 2) = diag(squeeze(mean(m_m_test)));
S(:, 2) = diag(squeeze(std(m_m_test)))/2;

M(:, 3) = diag(squeeze(mean(f_m_test)));
S(:, 3) = diag(squeeze(std(f_m_test)))/2;

M(:, 4) = diag(squeeze(mean(f_f_test)));
S(:, 4) = diag(squeeze(std(f_f_test)))/2;
M(:, 5) = diag(squeeze(mean(m_f_test)));
S(:, 5) = diag(squeeze(std(m_f_test)))/2;

M(:, 6) = diag(squeeze(mean(y_y_test)));
S(:, 6) = diag(squeeze(std(y_y_test)))/2;
M(:, 7) = diag(squeeze(mean(o_y_test)));
S(:, 7) = diag(squeeze(std(o_y_test)))/2;

M(:, 8) = diag(squeeze(mean(o_o_test)));
S(:, 8) = diag(squeeze(std(o_o_test)))/2;

M(:, 9) = diag(squeeze(mean(y_o_test)));
S(:, 9) = diag(squeeze(std(y_o_test)))/2;
figure;barwitherr(S,M);xlabel('wave', 'FontSize',14);legend({'all','m->m',...
    'f->m','f->f','m->f','y->y','o->y','o->o','y->o'}, 'FontSize',14);

ylabel('R2', 'FontSize',14);
figure;
feat_imp_w = plot_feat_stats(feat_imp_w, (feat_str), 0);
figure;
subplot(2,2,1);
feat_imp_m = plot_feat_stats(feat_imp_m, feat_str, 0);
title('Male');
subplot(2,2,2);
feat_imp_f = plot_feat_stats(feat_imp_f, feat_str, 0);
title('Female');
subplot(2,2,3);
feat_imp_y = plot_feat_stats(feat_imp_y, feat_str, 0);
setxlabels(feat_str);
title('Young');
subplot(2,2,4);
feat_imp_o = plot_feat_stats(feat_imp_o, feat_str, 0);
setxlabels((feat_str));legend('w1','w2','w3');
title('Old');
% 
% M=[];S=[];
% figure;
% for i = 1:3
%     M(:, 1) = squeeze(nanmean(feat_imp_w(:, i, :),3));
%     S(:, 1) = squeeze(nanstd(feat_imp_w(:, i, :), [], 3))/sqrt(2);
%     M(:, 2) = squeeze(nanmean(feat_imp_m(:, i, :),3));
%     S(:, 2) = squeeze(nanstd(feat_imp_m(:, i, :), [], 3))/sqrt(2);
%     M(:, 3) = squeeze(nanmean(feat_imp_f(:, i, :),3));
%     S(:, 3) = squeeze(nanstd(feat_imp_f(:, i, :), [], 3))/sqrt(2);
%     M(:, 4) = squeeze(nanmean(feat_imp_y(:, i, :),3));
%     S(:, 4) = squeeze(nanstd(feat_imp_y(:, i, :), [], 3))/sqrt(2);
%     M(:, 5) = squeeze(nanmean(feat_imp_o(:, i, :),3));
%     S(:, 5) = squeeze(nanstd(feat_imp_o(:, i, :), [], 3))/sqrt(2);
%     subplot(3,1,i);barwitherr(S,M);title(['wave ' num2str(i)]);
% end
% legend({'all','m','f','y','o'});
%     setxlabels(fliplr(feat_str));