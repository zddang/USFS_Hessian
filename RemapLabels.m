function newLabels = RemapLabels(labels)
    uniqueLabels = unique(labels);
    map = containers.Map(num2cell(uniqueLabels), num2cell(1:length(uniqueLabels)));
    newLabels = arrayfun(@(x) map(x), labels);
end
