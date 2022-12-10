# PolypNet
Lightweight Tkinter GUI designed to identify and segment polyps from colonoscopy images utilizing deep CNNs. PolypNet allows providers to upload screenshots of questionable colonoscopy images and outlights/highlights polyps found in the image. 

![image](https://user-images.githubusercontent.com/57613878/206839506-a626c92d-de94-41f4-9f77-4ac613c210d2.png)


This project is an extension from the research conducted below using the open source CVC-ClinicDB dataset containing colonoscopy images,
Jha D, Ali S, Tomar NK, Johansen HD, Johansen D, Rittscher J, Riegler MA, Halvorsen P. Real-Time Polyp Detection, Localization and Segmentation in Colonoscopy Using Deep Learning. IEEE Access. 2021 Mar 4;9:40496-40510. doi: 10.1109/ACCESS.2021.3063716. PMID: 33747684; PMCID: PMC7968127.

A UNet segmentaion model was implemented as seen below

![image](https://user-images.githubusercontent.com/57613878/206839536-397aff79-9b55-46f0-bccf-ef566da801db.png)

Images were one-hot encoded to represent background vs polyp classes as seen below

![image](https://user-images.githubusercontent.com/57613878/206874589-370810f3-df0e-4488-8c2b-08bcc955c9c4.png)

The IoU resulting from the 20 epochs of traning can also be seen below, steadying out around 0.97

![image](https://user-images.githubusercontent.com/57613878/206874683-0f3179ae-b284-4188-9180-a9a669e0ed51.png)

Testing data resulted in some very accuarte predictions as well as an increased amout of false positive classifications as seen below. Further studies can be conducted to tweak the UNet hyperparamaters to adjust for this increased FPR but at least the model is not missing any polyps which is the most imporant feature.

![image](https://user-images.githubusercontent.com/57613878/206874784-0d7c5d72-ebfe-45f7-a070-ed21ac27d6be.png)

