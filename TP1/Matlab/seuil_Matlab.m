close all;
clear;

%Ouverture d'une image au format couleur
ima=single(imread('../Image/ferrari.jpg'));
ima=ima./255;

%Affichage d'une image couleur avec image
figure('name','RGB in','numbertitle','off');image(ima);


%Taille d'une image
taille=size(ima);
display(taille);

ima_r=ima(:,:,1);
ima_g=ima(:,:,2);
ima_b=ima(:,:,3);

%Affichage d'un niveau de couleur de l'image 
##figure('name','R','numbertitle','off');imagesc(ima_r);colormap gray  %Niveau de rouge
##figure('name','G','numbertitle','off');imagesc(ima_g);colormap gray  %Niveau de vert
##figure('name','B','numbertitle','off');imagesc(ima_b);colormap gray  %Niveau de bleu

%Taille d'une image
[height, width, ~] = size(ima);
display(taille);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic toc pour mesurer le temps de calcul  
tic;  

ima_out=ima;

# isolate red
% Loop through each row
##for row = 1:height
##    % Loop through each column
##    for col = 1:width        
##        r = ima(row, col, 1);
##        g = ima(row, col, 2);
##        b = ima(row, col, 3);
##        nr = r/sqrt(r*r+g*g+b*b);
##        if nr <= 0.7  
##            ima_out(row,col,1)=0;
##            ima_out(row,col,2)=0;
##            ima_out(row,col,3)=0;
##        end
##
##    end
##end
#t = 22.5836 seconds

# change red to yellow
##for row = 1:height
##    % Loop through each column
##    for col = 1:width        
##        r = ima(row, col, 1);
##        g = ima(row, col, 2);
##        b = ima(row, col, 3);
##        nr = r/sqrt(r*r+g*g+b*b);
##        if nr > 0.7  
##            ima_out(row,col,1)=r;
##            ima_out(row,col,2)=r;
##            ima_out(row,col,3)=0;
##        end
##
##    end
##end
#t = 17.5671 seconds

#optimizes red
##conditions = ima(:, :, 1) ./ sqrt(sum(ima.^2, 3)) <= 0.7;
##ima_out(repmat(conditions, [1, 1, 3])) = 0;
#t = 0.020376 seconds


        

toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



figure('name','RGB out','numbertitle','off');image(ima_out);

%Sauvegarde d'une image au format jpg
imwrite(ima_out,'../Image/ferrari_out.jpg','jpg');



%Sauvegarde d'une image au format raw
fid = fopen('../Image/ferrari_out.raw', 'w');
fwrite(fid, ima_out, 'single');
fclose(fid);

