clear;
clc;
w=hamming(73)';
n=-36:1:36;
nh=1:1:36;
lp_filter_3040=[sin(2*pi*2665/16000.*(nh-37))./((nh-37)*pi),2*2665/16000,sin(2*pi*2665/16000.*nh)./(nh*pi)];
lp_filter_4960=[sin(2*pi*5335/16000.*(nh-37))./((nh-37)*pi),2*5335/16000,sin(2*pi*5335/16000.*nh)./(nh*pi)];
bp_filter=lp_filter_4960-lp_filter_3040;
lp_filter_2290=[sin(2*pi*2665/16000.*(nh-37))./((nh-37)*pi),2*2665/16000,sin(2*pi*2665/16000.*nh)./(nh*pi)];
hp_filter_5710=[-sin(2*pi*5335/16000.*(nh-37))./((nh-37)*pi),1-2*5335/16000,-sin(2*pi*5335/16000.*nh)./(nh*pi)];

% Window Function - Figure 1
figure(1);
subplot(2,1,1);
plot(n,abs(w));
grid on;
subplot(2,1,2);
plot(n,angle(w));
grid on;

% Sinc Functions for Two Ideal Low Pass Filters - Figure 2
figure(2);
subplot(2,1,1);
plot(n,lp_filter_3040);
title('low pass filter 3040');
grid on;
subplot(2,1,2);
plot(n,lp_filter_4960);
title('low pass filter 4960');
grid on;

% Magnitude Response of Band Pass Filter - Figure 3
figure(3);
plot((n+36)/73*16000,abs(fft(bp_filter.*w)));% plot(1:16000,abs(fft(bp_filter.*w,16000)));
grid on;

% Three Filters - Figure 4
figure(4);
plot(1:16000,abs(fft(lp_filter_2290.*w,16000)));
hold on;
plot(1:16000,abs(fft(hp_filter_5710.*w,16000)));
hold on;
plot(1:16000,abs(fft(bp_filter.*w,16000)));
grid on;

% Sound Play
sound_data=audioread('music16k.wav')';
sound([0.1*conv(sound_data(1,:),lp_filter_2290)+0.25*conv(sound_data(1,:),bp_filter)+2.65*conv(sound_data(1,:),hp_filter_5710);0.1*conv(sound_data(2,:),lp_filter_2290)+0.25*conv(sound_data(2,:),bp_filter)+2.65*conv(sound_data(2,:),hp_filter_5710)],16000);
