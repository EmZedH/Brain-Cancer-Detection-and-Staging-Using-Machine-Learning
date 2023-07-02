clc
close all
clear all

test_image = imread('C:\Users\omrst\Downloads\Project\train image\yes\Y1.jpg');

img_folder = ('C:\Users\omrst\Downloads\Project\train image\train image\yes'); 
filenames1 = dir(fullfile(img_folder,'*.jpg'));
filenames2 = dir(fullfile(img_folder,'*.png'));
Total_image = numel(filenames1)+numel(filenames2);
for i=1:Total_image
    if i<=numel(filenames1)
    
        f = fullfile(img_folder,filenames1(i).name); 
        Output = imread(f);
        try
            f = rgb2gray(Output);
        catch
            f = Output;
        end
        arr{i,1} = f; 
    else
         f = fullfile(img_folder,filenames2(Total_image-i+1).name); 
        Output = imread(f); 
        try
            f = rgb2gray(Output);
        catch
            f = Output;
        end
        arr{i,1} = f;
    end
    
end

df=[];
Gt=[];
for i=1:Total_image
    r=arr{i,1};
    

    [rw,cl]=size(r);
    
    b=medfilt2(r,[10 10]) ;
    

    [y,x]=imhist(b) ;
    u=x.*y;
    v=x.*x;
    w=y.*v;
    n=sum(y) ;
    mean=sum(u)/sum(y);
    var=sum(w)/sum(y)-mean*mean;
    std=(var)^0.5;
    t=mean+0.5*std;
    bw=b>t;

    
    mk=imhmin(255-uint8(bwdist(~bw)),2);

    gs=watershed(mk);

    abs=gs==0;

    w=labeloverlay(double(bw),double(abs),'colormap',[1 1 1],'Transparency',0);

    w1=imerode(rgb2gray(w),strel('arbitrary',20));
    bm=bwareaopen(w1,2000);

    fi=w1;
    fi(bm)=false;

    er=bwareaopen(fi,55);

    li=bwlabel(er,8);
    rm=regionprops(li,b,'all');
    ecc=[rm.Eccentricity];
    rn=size(rm,1);
    aei=(ecc<0.98);
    ki=find(aei);
    ri=ismember(li,ki);
    er=ri;

    clear li;
    clear rm;
    clear rn;
    li=bwlabel(er,8);
    rm=regionprops(li,b,'all');

    rn=size(rm,1);

    for k=1:rn
        s=0;
        ra=rm(k).Area;
        rp=rm(k).Perimeter;
        rc=rm(k).Centroid;
        rd(k)=sqrt(4*ra/pi);
        re=rm(k).Eccentricity;
        r=floor(rd(k)/2);

        if r<3
            s=1;
        elseif r>3&r<7
            s=2;
        elseif r>7&r<10
            s=3;
        else
            s=4;
        end
        
        Fr=[ra,rp,rc,rd(k),re];
        df=[df;Fr];
        Gt=[Gt;s];

    end
    
end

try
    c=rgb2gray(test_image);
catch
    c=test_image;
end

Testftr=[];
figure
f1=12;
imshow(c);
title('Original image');
[rw,cl]=size(c)

b1=medfilt2(c,[10 10]) ;

figure
imshow(histeq(b1));
title('filter')

[y,x]=imhist(b1) ;
u=x.*y;
v=x.*x;
w=y.*v;
n=sum(y) ;
mean1=sum(u)/sum(y);
var=sum(w)/sum(y)-mean1*mean1;
std=(var)^0.5;
t=mean1+0.5*std;
bw1=b1>t;
figure
imshow(bw1)
title('binary')
mk1=imhmin(255-uint8(bwdist(~bw1)),2);
figure;
imshow(mk1,[]);
title('distance transformed image');
gs1=watershed(mk1);
figure;
imshow(label2rgb(gs1));
title('watershed');
abs1=gs1==0;
figure;
w1=labeloverlay(double(bw1),double(abs1),'colormap',[1 1 1],'Transparency',0);
imshow(w1);
title('segmented image');
w11=imerode(im2gray(w1),strel('arbitrary',20));
bm1=bwareaopen(w11,1500);
figure
imshow(bm1)
title('big mask')
fi1=w11;
fi1(bm1)=false;
er1=bwareaopen(fi1,55);
li1=bwlabel(er1,8);
rm1=regionprops(li1,b1,'all');
ecc1=[rm1.Eccentricity];
rn1=size(rm1,1);
aei1=(ecc1<0.98);
ki1=find(aei1);
ri1=ismember(li1,ki1);
er1=ri1;

li1=bwlabel(er1,8);
rm1=regionprops(li1,b1,'all');

boundaries1=bwboundaries(er1);
nob1=size(boundaries1,1);
for k1=1:nob1
    thisboundary1=boundaries1{k1};

end
hold off;
rm1=regionprops(li1,b1,'all');
rn1=size(rm1,1);
t1=14;
l1=-7;
recd=zeros(1,rn1);
% fprintf(1,'region   area  perimeter   centroid      diameter  Eccentricity Orientation\n');
% for k11=1:rn1
%     tf=[];
%     ra1=rm1(k11).Area;
%     ro1=rm1(k11).Orientation;
%     rp1=rm1(k11).Perimeter;
%     rc1=rm1(k11).Centroid;
%     rd1(k11)=sqrt(4*ra1/pi);
%     re1=rm1(k11).Eccentricity;
%     r1=floor(rd1(k11)/2);
%     fprintf(1,'#%2d%11.1f%8.1f%8.1f%8.1f%8.1f%8.1f        %8.1f\n',k11,ra1,rp1,rc1,rd1(k11),re1,ro1);
%     text(rc1(1)+l1,rc1(2),num2str(k11),'FontSize',t1,'FontWeight','Bold');
%     tf=horzcat([ra1,rp1,rc1,rd1(k11),re1]);
%     Testftr=[Testftr;tf];
% 
% end
Trainingset=df;

length=numel(Gt);
GroupTrain=cell(1,length);
for i=1:length
    a=int2str(Gt(i));
    GroupTrain{1,i}=a;
end
Y=GroupTrain;
classes=unique(Y);
SVMmodels=cell(numel(classes),1);
rng(1);
for j=1:numel(classes)
    idx=strcmp(Y',classes(j));
    SVMmodels{j}=fitcsvm(df,idx,'ClassNames',[false true],'Standardize',true,'KernelFunction','rbf','BoxConstraint',1);
end
[rows, col]=size(Testftr);

num_iter = 10;
    delta_t = 1/7;
    kappa = 15;
    option = 2;

    inp = anisodiff(test_image,num_iter,delta_t,kappa,option);
    inp = uint8(inp);
    
inp=imresize(inp,[256,256]);
if size(inp,3)>1
    inp=rgb2gray(inp);
end

sout=imresize(inp,[256,256]);
t0=mean(test_image(:));
th=t0+((max(inp(:))+min(inp(:)))./2);
for i=1:1:size(inp,1)
    for j=1:1:size(inp,2)
        if inp(i,j)>th
            sout(i,j)=1;
        else
            sout(i,j)=0;
        end
    end
end



label=bwlabel(sout);
stats=regionprops(logical(sout),'Solidity','Area','BoundingBox');
density=[stats.Solidity];
area=[stats.Area];
high_dense_area=density>0.7;
max_area=max(area(high_dense_area));
tumor_label=find(area==max_area);
tumor=ismember(label,tumor_label);


if max_area>80

    if max_area < 200
        fprintf('\n\nBenign Tumor')
    elseif max_area >200 && max_area < 500
        fprintf('\n\nStage 1');
    elseif max_area > 500 && max_area < 1500
        fprintf('\n\nStage 2');
    elseif max_area > 1500 && max_area < 2500
        fprintf('\n\nStage 3');
    else
        fprintf('\n\nStage 4')
    end

   figure;
   imshow(tumor)
   title('tumor alone','FontSize',20);
else

    fprintf('\n\nNo Tumor');

end
            

function diff_im = anisodiff(im, num_iter, delta_t, kappa, option)
im = double(im);
diff_im = im;

dx = 1;
dy = 1;
dd = sqrt(2);

hN = [0 1 0; 0 -1 0; 0 0 0];
hS = [0 0 0; 0 -1 0; 0 1 0];
hE = [0 0 0; 0 -1 1; 0 0 0];
hW = [0 0 0; 1 -1 0; 0 0 0];
hNE = [0 0 1; 0 -1 0; 0 0 0];
hSE = [0 0 0; 0 -1 0; 0 0 1];
hSW = [0 0 0; 0 -1 0; 1 0 0];
hNW = [1 0 0; 0 -1 0; 0 0 0];

for t = 1:num_iter

        nablaN = imfilter(diff_im,hN,'conv');
        nablaS = imfilter(diff_im,hS,'conv');   
        nablaW = imfilter(diff_im,hW,'conv');
        nablaE = imfilter(diff_im,hE,'conv');   
        nablaNE = imfilter(diff_im,hNE,'conv');
        nablaSE = imfilter(diff_im,hSE,'conv');   
        nablaSW = imfilter(diff_im,hSW,'conv');
        nablaNW = imfilter(diff_im,hNW,'conv'); 
        
        if option == 1
            cN = exp(-(nablaN/kappa).^2);
            cS = exp(-(nablaS/kappa).^2);
            cW = exp(-(nablaW/kappa).^2);
            cE = exp(-(nablaE/kappa).^2);
            cNE = exp(-(nablaNE/kappa).^2);
            cSE = exp(-(nablaSE/kappa).^2);
            cSW = exp(-(nablaSW/kappa).^2);
            cNW = exp(-(nablaNW/kappa).^2);
        elseif option == 2
            cN = 1./(1 + (nablaN/kappa).^2);
            cS = 1./(1 + (nablaS/kappa).^2);
            cW = 1./(1 + (nablaW/kappa).^2);
            cE = 1./(1 + (nablaE/kappa).^2);
            cNE = 1./(1 + (nablaNE/kappa).^2);
            cSE = 1./(1 + (nablaSE/kappa).^2);
            cSW = 1./(1 + (nablaSW/kappa).^2);
            cNW = 1./(1 + (nablaNW/kappa).^2);
        end

        diff_im = diff_im + ...
                  delta_t*(...
                  (1/(dy^2))*cN.*nablaN + (1/(dy^2))*cS.*nablaS + ...
                  (1/(dx^2))*cW.*nablaW + (1/(dx^2))*cE.*nablaE + ...
                  (1/(dd^2))*cNE.*nablaNE + (1/(dd^2))*cSE.*nablaSE + ...
                  (1/(dd^2))*cSW.*nablaSW + (1/(dd^2))*cNW.*nablaNW );
           
       
        
end
end
