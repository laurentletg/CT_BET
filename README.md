# ATTENTION! 
### Make sure to create an environnement with the following requirements to save some time and trouble shooting: 
tensorflow                1.15.0;
keras                     2.3.1;
h5py                      2.10.0;

the following packages can probably be updated up to date, but to be sure, those are compatible versions

simpleitk                 2.3.0;
scikit-learn              1.0.2;
scipy                     1.7.3;
nibabel                   4.0.2;
numpy                     1.21.6;

### quick start
1. Downloads the weight files from google drive link in the weight_file in the weights folder and place it in the weights_folder
   (IF YOU WANT TO USE A PRETRAINED MODEL, DROP YOUR OWN WEIGHT FILE IN THE FOLDER)
2. Drop your images under image_data
3. set predict flag to be true in unet_CT_SS.py file and "testLabelFlag" to be False
4. In the main function at the bottom of unet_CT_SS.py, validate that weightFile has the correct path ('unet_CT_SS_2017... .h5' if you're the 2d model (Predicti), and 'unet_CT_SS_3D_...' if you're using Predict3D, or you're own trained weight)

5. To use the 3d model, you just have to uncomment #unetSS.Predict3D(weightFile) and comment out unetSS.Predict(weightFile)

Note. If you're having a Value Error for some files, remove one by one the problematic files and put them in a separate directory. At the end, once the inference has been done correctly with all the files that dont show up a ValueError, switch the images in the image_data folder for the problematic ones, and uncomment the last bloc in load3Ddata.py. You should be able to run inference on those few images without problem (BUT DO NOT USE FOR IMAGES THAT DON'T NEED IT) You might need to binarize those few images

6. Run unet_CT_SS.py

Original Readme below is also a good reference for deeper usage
link to original CT_BET Repo : https://github.com/aqqush/CT_BET










# CT_BET Original repo README
## CT_BET: Robust Brain Extraction Tool for CT Head Images
To be able to run this you will need to have a GPU card available and have tensorflow and keras libraries installed in python.

In order to train the model on CT head images
1) clone the repository to your local drive. 
2) Put your images under 'image_folder' and masks under 'mask_folder'
3) run unet_CT_SS.py

In order to extract brain from a CT head image.
1) clone the repository to your local drive.
2) put your images under image_folder
3) set predict flag to be true in unet_CT_SS.py file
4) make sure you set "testLabelFlag" to be False, in the unet_CT_SS.py
5) run unet_CT_SS.py

Additionals:
==========================================
1)If you want to run the model on a new data without mask, you should set "testLabelFlag=False", which computes the DICE metric if you have masks in the mask_data folder. If you run metrics "testLabelFlag=True" on your new data with mask, make sure that you have them both with the same name in their folders.

2)You should also download the model weights into the weights_folder as instructed in text file within the weights_folder.

==========================================

Please contact me if you have any difficulty in running this code.

Email: akkus.zeynettin@mayo.edu

Zeynettin Akkus

Please cite the paper below if you use the tool in your study:
1) Zeynettin Akkus, Petro M. Kostandy, Kenneth A. Philbrick, Bradley J. Erickson. Proceedings Volume 10574, Medical Imaging 2018: Image Processing; 1057420 (2018) https://doi.org/10.1117/12.2293423

2) Zeynettin Akkus, Petro M. Kostandy, Kenneth A. Philbrick, Bradley J. Erickson. Robust Brain Extraction Tool for CT Head Images. In Press. Neurocomputing.https://doi.org/10.1016/j.neucom.2018.12.085
