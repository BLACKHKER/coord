m = cameraParams.IntrinsicMatrix;
k = cameraParams.RadialDistortion;
p = cameraParams.TangentialDistortion;
%k = [0,0,0];
%p = [0,0];

% 没勾选三个切向参数，补齐
if size(k, 2) == 2
    k = [k,0];
end

p = [p,0];
m = [m;k;p];
writematrix(m, '../csv/MATLAB_Camera_Intrinsics.csv')