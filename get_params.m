m = cameraParams.IntrinsicMatrix';
k = cameraParams.RadialDistortion;
p = cameraParams.TangentialDistortion;
%k = [0,0,0];
%p = [0,0];
if size(k, 2) == 2
    k = [k,0];
end
p = [p,0];
m = [m;k;p];
writematrix(m, 'params.csv')