function [results] = copmute_dct(images) 
%matlab script for converting an image(gray) to its dct

if ndims(images) < 3
    error('Number of Images should be greter than 1')
end

results = zeros([size(images,1), size(images,2), size(images,3)]);
for i=1:size(images,1)

    if ndims(images )== 4
        results(:,:,i) = dct2(rgb2gray(images(:,:,:,i)));
    else 
        results(:,:,i) = dct2(images(:,:,i));
    end

end

end