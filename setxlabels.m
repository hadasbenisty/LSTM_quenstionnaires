function setxlabels(str)
for i = 1:length(str)
    str{i}(str{i} == '_') = ' ';
end
set(gca, 'XTick', 1:length(str));
set(gca, 'XTickLabel', str);
set(gca, 'XTickLabelRotation', 90);
set(gca, 'FontSize', 14);
