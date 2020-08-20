function []  = LCS_ellipse()
%% parameters illustration
%1) Tac: 
%The threshold of elliptic angular coverage which ranges from 0~360. 
%The higher Tac, the more complete the detected ellipse should be.
%2) Tr:
%The ratio of support inliers to ellipse which ranges from 0~1.
%The higher Tr, the more sufficient the support inliers are.
%3) specified_polarity: 
%1 means detecting the ellipses with positive polarity;
%-1 means detecting the ellipses with negative polarity; 
%0 means detecting all ellipses from image


close all;

%image path
filename = 'D:\Graduate Design\Ellipse Detection\MyEllipse - github\pics\666.jpg';

% parameters
Tac = 165;
Tr = 0.6;
specified_polarity = 0;

%%
% read image 
disp('------read image------');
I = imread(filename);


%% detecting ellipses from real-world images
[ellipses, ~, posi] = ellipseDetectionByArcSupportLSs(I, Tac, Tr, specified_polarity);

disp('draw detected ellipses');
drawEllipses(ellipses',I);
% display
ellipses(:,5) = ellipses(:,5)./pi*180;
ellipses
disp(['The total number of detected ellipses£º',num2str(size(ellipses,1))]);

%% draw ellipse centers
%hold on;
%candidates_xy = round(posi+0.5);%candidates' centers (col_i, row_i)
%plot(candidates_xy(:,1),candidates_xy(:,2),'.');%draw candidates' centers.

%% write the result image
%set(gcf,'position',[0 0 size(I,2) size(I,1)]);
%saveas(gcf, 'D:\Graduate Design\Ellipse Detection\MyEllipse - github\pics\666_all.jpg', 'jpg');
end



