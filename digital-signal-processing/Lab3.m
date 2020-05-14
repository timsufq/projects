% % Q1
% clear;
% fs=10000;
% num=[1,2,1];
% den=[1,-1.25,0.78125];
% % zplane(num,den);
% freqz(num,den,fs);

% Q2
clear;
white_noise_data=audioread('white.wav');
pink_noise_data=audioread('pink.wav');
babble_noise_data=audioread('babble.wav');
f16_noise_data=audioread('f16.wav');
music_data=audioread('music16k.wav');
% sound(white_noise_data,16000);
% sound(pink_noise_data,16000);
% sound(babble_noise_data,16000);
% sound(f16_noise_data,16000);
% % clear playsnd

figure(1);
plot(0:1:2046,xcorr(white_noise_data(1:1024)'),'b');
hold on;
plot(0:1:2046,xcorr(pink_noise_data(1:1024)'),'r');
legend('white noise','pink noise');

lr_channel_cc=xcorr(music_data(10001:11024,1),music_data(10001:11024,2),'coeff');
lb_channel_cc=xcorr(music_data(10001:11024,1),babble_noise_data(10001:11024),'coeff');
lrw_channel_cc=xcorr(music_data(10001:11024,1),music_data(10001:11024,2)+white_noise_data(10001:11024),'coeff');
% figure(1);
% plot(lr_channel_cc);
% title('Left and Right Channel');
% figure(2);
% plot(lb_channel_cc);
% title('Left Channel and Babble');
% figure(3);
% plot(lrw_channel_cc);
% title('Left Channel and Right Channel with White Noise');
% figure(4);
% plot(lr_channel_cc,'b');
% hold on;
% plot(lrw_channel_cc,'r');
% legend('Left and Right Channel','Left Channel and Right Channel with White Noise');
% title('Comparasion between the Left and Right Channel Plot and the Left Channel and Right Channel with White Noise Plot');
% figure(1);
% plot(lr_channel_cc,'b');
% hold on;
% plot(lb_channel_cc,'r');
% hold on;
% plot(lrw_channel_cc,'y');
% legend('Left and Right Channel','Left Channel and Babble','Left Channel and Right Channel with White Noise');
% title('Combined');

% figure(1);
% plot(lr_channel_cc);
% hold on;
% plot(lb_channel_cc);
% legend('Left and Right Channel','Left Channel and Babble');
% title('The Left and Right Channel Plot and the Left Channel and Babble Plot');