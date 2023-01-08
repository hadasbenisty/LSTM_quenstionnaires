function feat_imp_w = plot_feat_stats(feat_imp_w, feat_str, tonorm)
feat_imp_w = permute(feat_imp_w, [2 3 1 4]);
feat_imp_w = reshape(feat_imp_w, size(feat_imp_w, 1), 3, []);

for i = 1:size(feat_imp_w, 2)
    [~, p(:, i)] = ttest(squeeze(feat_imp_w(:, i, :, :))', 0);
end
if tonorm
    feat_imp_w = abs(feat_imp_w);
    
    feat_imp_w = bsxfun(@rdivide, feat_imp_w, nansum(feat_imp_w));
end
M = squeeze(nanmean(feat_imp_w, 3));
S = squeeze(nanstd(feat_imp_w, [], 3))/sqrt(sum(~isnan(squeeze(feat_imp_w(1,1,:))))-1);
M = M(2:end, :);
S = S(2:end, :);
feat_str = feat_str(2:end);
b=barwitherr(S(2:2:end,:),M(2:2:end,:));
setxlabels(feat_str(2:2:end));
% hold all
% for i = 1:3
%     y = b(i).YEndPoints;
%     x = b(i).XEndPoints;
%     inds = find(p(:, i) < 0.01);
%     plot(x(inds), y(inds)+0.05, 'k*')
% end
        

