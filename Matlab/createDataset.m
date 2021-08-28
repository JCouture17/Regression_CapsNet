%% Init
clear; clc;
run('LoadData.m');
data = struct2table(batch_combined);
cl = data.cycle_life;  
    % We're treating batteries with cycle life >1175 and <400 as outliers
    % We're losing 11 batteries to outlying tendencies
for i = 1:length(cl)
    if data.cycle_life(i) < 400 || data.cycle_life(i) > 1175
        data.cycle_life(i) = 0;
    end    
end
data = data(data.cycle_life ~= 0, :);

data.cycle_life([37:39, 42:48, 50, 53:56, 59]) = 0;
data = data(data.cycle_life ~= 0,:);

summary = data.summary;

%% TODO
% Remove half of the batteries with ~400 cycle life to obtain a consistant
% variance
% Could then use battery 1 for testing (cycle life = 1074)

%% Curve Smoothing
% Discharging Capacity
for i = 1:height(summary)
    b = summary(i,:).QDischarge;
    b = filloutliers(b, 'linear', 'movmedian', 5);
    b = smoothdata(b);
    summary(i,:).QDischarge = b;
end

% Charging Capacity
for i = 1:height(summary)
    b = summary(i,:).QCharge;
    b = filloutliers(b, 'linear', 'movmedian', 5);
    b = smoothdata(b);
    summary(i,:).QCharge = b;
end

% Internal Resistance
for i = 1:height(summary)
    a = summary(i,:).IR;
    a = filloutliers(a, 'linear', 'movmedian', 10);
    a = smoothdata(a);
    summary(i,:).IR = a;
end
    
% Current Curves
cycles = data.cycles;
for i = 1:length(cycles)
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).I;
        b = filloutliers(b, 'linear', 'movmedian', 50);
        b = smooth(b);
        a(j).I = b;
    end
end

% Voltage Curves
for i = 1:length(cycles)
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).V;
        b = smooth(b);
        a(j).V = b;
    end
end

% Temperature Curves
for i = 1:length(cycles)
    a = cell2mat(cycles(i));
    for j = 2:length(a)
        b = a(j).T;
        b = smooth(b);
        a(j).T = b;
    end
end  

%% Image as Inputs
cycle = 10;
rul = [];
folder = 'D:\Toyota\Full_data.tif';
rulFileName = 'D:\Toyota\Full_rul.mat';
picRes = 128; % Image dimensions, in pixels x pixels

for i = 1:99 % For each battery
    cap = normalize(summary(i,:).QCharge, 'range', [0 5]);
    ir = normalize(summary(i,:).IR*100, 'range');
    a = cell2mat(data.cycles(i));
    c = flip(summary(i,:).cycle);
    for j = cycle+1:length(a)-cycle % For each cycle of each battery
%         close all
        recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
        I = [a(j).I(1:recording_length) a(j+cycle-1).I(1:recording_length)];
%         V = [a(j).V(1:recorded_length) a(j+cycle-1).V(1:recorded_length)];
%         T = [a(j).T(1:recorded_length) a(j+cycle-1).T(1:recorded_length)];
        Qc = [a(j).Qc(1:recording_length) a(j+cycle-1).Qc(1:recording_length)];
        Qd = [a(j).Qd(1:recording_length) a(j+cycle-1).Qd(1:recording_length)];
        cap1 = cap(j+cycle-1);
        ir1 = ir(j+cycle-1);
        rul = [rul; c(j+cycle)];
        
        fig = figure('Position', [680 558 picRes picRes], 'visible', 'off');
        subplot(2,2,1, 'Position', [0 0.5 0.5 0.5]); hold on; 
        plot(I(:,1), 'Color', 'r'); 
        plot(I(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [-7 7], 'XLim', [0 1000]);
        subplot(2,2,2, 'Position', [0.5 0.5 0.5 0.5]); hold on; 
        plot(Qc(:,1), 'Color', 'r'); 
        plot(Qc(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,3, 'Position', [0 0 0.5 0.5]); hold on; 
        plot(Qd(:,1), 'Color', 'r'); 
        plot(Qd(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,4, 'Position', [0.5 0 0.5 0.5]); 
        t = text(0.2, 0.7, sprintf('%.4f',cap1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        t = text(0.2, 0.2, sprintf('%.4f',ir1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        
        f = getframe(gcf); new = f.cdata;
        if j == cycle+1 && i == 1
            imwrite(new, folder);
        else
            imwrite(new, folder, 'WriteMode', 'append');
        end
    end
end

%% 5 Cycles
cycle = 5;
for i = 1:99 % For each battery
    cap = normalize(summary(i,:).QCharge, 'range', [0 5]);
    ir = normalize(summary(i,:).IR*100, 'range');
    a = cell2mat(data.cycles(i));
    c = flip(summary(i,:).cycle);
    for j = cycle+1:length(a)-cycle % For each cycle of each battery
        recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
        I = [a(j).I(1:recording_length) a(j+cycle-1).I(1:recording_length)];
        Qc = [a(j).Qc(1:recording_length) a(j+cycle-1).Qc(1:recording_length)];
        Qd = [a(j).Qd(1:recording_length) a(j+cycle-1).Qd(1:recording_length)];
        cap1 = cap(j+cycle-1);
        ir1 = ir(j+cycle-1);
        rul = [rul; c(j+cycle)];
        
        fig = figure('Position', [680 558 picRes picRes], 'visible', 'off');
        subplot(2,2,1, 'Position', [0 0.5 0.5 0.5]); hold on; 
        plot(I(:,1), 'Color', 'r'); 
        plot(I(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [-7 7], 'XLim', [0 1000]);
        subplot(2,2,2, 'Position', [0.5 0.5 0.5 0.5]); hold on; 
        plot(Qc(:,1), 'Color', 'r'); 
        plot(Qc(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,3, 'Position', [0 0 0.5 0.5]); hold on; 
        plot(Qd(:,1), 'Color', 'r'); 
        plot(Qd(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,4, 'Position', [0.5 0 0.5 0.5]); 
        t = text(0.2, 0.7, sprintf('%.4f',cap1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        t = text(0.2, 0.2, sprintf('%.4f',ir1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        
        f = getframe(gcf); new = f.cdata;
        imwrite(new, folder, 'WriteMode', 'append');
    end
end

%% 3 Cycles
cycle = 3;
for i = 1:99 % For each battery
    cap = normalize(summary(i,:).QCharge, 'range', [0 5]);
    ir = normalize(summary(i,:).IR*100, 'range');
    a = cell2mat(data.cycles(i));
    c = flip(summary(i,:).cycle);
    for j = cycle+1:length(a)-cycle % For each cycle of each battery
        recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
        I = [a(j).I(1:recording_length) a(j+cycle-1).I(1:recording_length)];
        Qc = [a(j).Qc(1:recording_length) a(j+cycle-1).Qc(1:recording_length)];
        Qd = [a(j).Qd(1:recording_length) a(j+cycle-1).Qd(1:recording_length)];
        cap1 = cap(j+cycle-1);
        ir1 = ir(j+cycle-1);
        rul = [rul; c(j+cycle)];
        
        fig = figure('Position', [680 558 picRes picRes], 'visible', 'off');
        subplot(2,2,1, 'Position', [0 0.5 0.5 0.5]); hold on; 
        plot(I(:,1), 'Color', 'r'); 
        plot(I(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [-7 7], 'XLim', [0 1000]);
        subplot(2,2,2, 'Position', [0.5 0.5 0.5 0.5]); hold on; 
        plot(Qc(:,1), 'Color', 'r'); 
        plot(Qc(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,3, 'Position', [0 0 0.5 0.5]); hold on; 
        plot(Qd(:,1), 'Color', 'r'); 
        plot(Qd(:,2), 'Color', 'b'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,4, 'Position', [0.5 0 0.5 0.5]); 
        t = text(0.2, 0.7, sprintf('%.4f',cap1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        t = text(0.2, 0.2, sprintf('%.4f',ir1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        
        f = getframe(gcf); new = f.cdata;
        imwrite(new, folder, 'WriteMode', 'append');
    end
end

%% 1 Cycle
cycle = 1;
for i = 1:99
    cap = normalize(summary(i,:).QCharge, 'range', [0 5]);
    ir = normalize(summary(i,:).IR*100, 'range');
    a = cell2mat(data.cycles(i));
    c = flip(summary(i,:).cycle);
    for j = cycle+1:length(a)-cycle
        recording_length = min(length(a(j).I), length(a(j+cycle-1).I));
        curr = a(j).I(1:recording_length);
        V = a(j).V(1:recording_length);
        T = a(j).T(1:recording_length);
        Qc = a(j).Qc(1:recording_length);
        Qd = a(j).Qd(1:recording_length);
        cap1 = cap(j+cycle-1);
        ir1 = ir(j+cycle-1);
        rul = [rul; c(j+cycle)];
        
        fig = figure('Position', [680 558 picRes picRes], 'visible', 'off');
        subplot(2,2,1, 'Position', [0 0.5 0.5 0.5]); hold on; 
        plot(curr(:,1), 'Color', 'r');  
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [-7 7], 'XLim', [0 1000]);
        subplot(2,2,2, 'Position', [0.5 0.5 0.5 0.5]); hold on; 
        plot(Qc(:,1), 'Color', 'r');  
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,3, 'Position', [0 0 0.5 0.5]); hold on; 
        plot(Qd(:,1), 'Color', 'r'); 
        set(gca, 'YTickLabel', [], 'XTickLabel', [],...
            'YLim', [0 1.2], 'XLim', [0 1000]);
        subplot(2,2,4, 'Position', [0.5 0 0.5 0.5]); 
        t = text(0.2, 0.7, sprintf('%.4f',cap1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        t = text(0.2, 0.2, sprintf('%.4f',ir1)); 
        t.FontSize=10; set(gca, 'visible', 'off');
        
        f = getframe(gcf); new = f.cdata;
        imwrite(new, folder, 'WriteMode', 'append');
        end
end

save(rulFileName, 'rul');
