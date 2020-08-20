function [ellipse,info] = fitEllipse(X,Y)
% FITELLIPSE Least-squares fit of ellipse to 2D points.
% A = FITELLIPSE(X,Y) returns the parameters of the best-fit
% ellipse to 2D points (X,Y).
% The returned vector A contains the center, radii, and orientation
% of the ellipse, stored as (Cx, Cy, Rx, Ry, theta_radians)


%{
% normalize data
mx = mean(X);
my = mean(Y);
sx = (max(X)-min(X))/2;
sy = (max(Y)-min(Y))/2;
x = (X-mx)/sx;
y = (Y-my)/sy;
% Force to column vectors
x = x(:);
y = y(:);
%}

x = X(:);%change session 1
y = Y(:);
% Build design matrix
D = [ x.*x x.*y y.*y x y ones(size(x)) ];
% Build scatter matrix
S = D'*D;
% Build 6x6 constraint matrix
C(6,6) = 0; C(1,3) = 2; C(2,2) = -1; C(3,1) = 2;
% Solve eigensystem
[gevec, geval] = eig(S,C);
% Find the positive eigenvalue
I = find(real(diag(geval)) > -1e-8 & ~isinf(diag(geval)));
if( isempty(I) )
    info = 0;
    ellipse = [];
    return;
end
% Extract eigenvector corresponding to negative eigenvalue
A = real(gevec(:,I));
% unnormalize
%{
par = [
A(1)*sy*sy, ...
A(2)*sx*sy, ...
A(3)*sx*sx, ...
-2*A(1)*sy*sy*mx - A(2)*sx*sy*my + A(4)*sx*sy*sy, ...
-A(2)*sx*sy*mx - 2*A(3)*sx*sx*my + A(5)*sx*sx*sy, ...
A(1)*sy*sy*mx*mx + A(2)*sx*sy*mx*my + A(3)*sx*sx*my*my ...
- A(4)*sx*sy*sy*mx - A(5)*sx*sx*sy*my ...
+ A(6)*sx*sx*sy*sy ...
]';
%}

par = [A(1),A(2),A(3),A(4),A(5),A(6)]; %change session 2
% Convert to geometric radii, and centers
thetarad = 0.5*atan2(par(2),par(1) - par(3));

cost = cos(thetarad);
sint = sin(thetarad);
sin_squared = sint.*sint;
cos_squared = cost.*cost;
cos_sin = sint .* cost;
Ao = par(6);
Au = par(4) .* cost + par(5) .* sint;
Av = - par(4) .* sint + par(5) .* cost;
Auu = par(1) .* cos_squared + par(3) .* sin_squared + par(2) .* cos_sin;
Avv = par(1) .* sin_squared + par(3) .* cos_squared - par(2) .* cos_sin;
% ROTATED = [Ao Au Av Auu Avv]
tuCentre = - Au./(2.*Auu);
tvCentre = - Av./(2.*Avv);
wCentre = Ao - Auu.*tuCentre.*tuCentre - Avv.*tvCentre.*tvCentre;
uCentre = tuCentre .* cost - tvCentre .* sint;
vCentre = tuCentre .* sint + tvCentre .* cost;
Ru = -wCentre./Auu;
Rv = -wCentre./Avv;
% Ru = sqrt(abs(Ru)).*sign(Ru);
% Rv = sqrt(abs(Rv)).*sign(Rv);
Ru = sqrt(abs(Ru));
Rv = sqrt(abs(Rv));

%code by Alan Lu
% delta   = 4.*par(1).*par(3)-par(2).*par(2);
% uCentre = (par(2).*par(5)-2.*par(3).*par(4))./delta;
% vCentre = (par(2).*par(4)-2.*par(1).*par(5))./delta;
% temp1 = 2.*(par(1).*uCentre.*uCentre+par(3).*vCentre.*vCentre+par(2).*uCentre.*vCentre-par(6));
% temp2 = par(1)+par(3);
% temp3 = sqrt((par(1)-par(3)).*(par(1)+par(3))+par(2).*par(2));
% Ru = temp1./(temp2+temp3);
% Rv = temp1./(temp2-temp3);

info = 1;%return successfully
ellipse = [uCentre, vCentre, Ru, Rv, thetarad];%x0(col) y0(row) semimajor semiminor thera_rad
 %会出现Ru < Rv情况，对调一下
if(Ru < Rv )
   ellipse(3) = Rv;
   ellipse(4) = Ru;
   if(thetarad < 0)
     ellipse(5) = ellipse(5)+1.570796326794897; %pi/2
   else
     ellipse(5) = ellipse(5)-1.570796326794897;
   end
end
end

%draw ellipse
% hold on;
% th=0:pi/180:2*pi;
% Semi_major= Ru;
% Semi_minor= Rv;
% x0= uCentre;
% y0= vCentre;
% Phi= thetarad;
% x=x0+Semi_major*cos(Phi)*cos(th)-Semi_minor*sin(Phi)*sin(th);
% y=y0+Semi_minor*cos(Phi)*sin(th)+Semi_major*sin(Phi)*cos(th);    
% plot(x,y,'r', 'LineWidth',1);

