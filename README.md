<H1 CLASS="western" style="text-align:center;">ZZDeepRollover</H1>

This code enables the detection of rollovers performed by zebrafish larvae tracked by the open-source software <a href="https://github.com/oliviermirat/ZebraZoom" target="_blank">ZebraZoom</a>. This code is still in "beta mode". For more information visit <a href="https://zebrazoom.org/" target="_blank">zebrazoom.org</a> or email us at info@zebrazoom.org<br/>

<H2 CLASS="western">Basic Usage:</H2>
You must first install tensorflow 1.15.0, tensorflow-hub 0.12.0 and keras 2.7.0 on your machine. It may be better to first create an anaconda environment for this purpose. ZZDeepRollover has been tested with Python 3.7 and may not work with other versions of Python (it won't work with Python 3.9 for example).<br/>
Then download <a href="https://drive.google.com/drive/folders/11alx_zUQylt5Xf4OvxN6csciLL3IPLOO?usp=sharing" target="_blank">this model</a> and place all 4 files downloaded inside the "model" subfolder of this repo.<br/>
Then, to run the rollover detection on ZebraZoom tracking results, type:<br/><br/>
python detectRolloverFramesWithNewZZversion.py pathToZZoutputFolder videoToAnalyzeFolderName pathToOriginalVideo<br/><br/>
The rollover detection results will be saved inside the video result folder.<br/>


<H2 CLASS="western">Going further:</H2>

[Preparing the rollovers detection model](#preparing)<br/>
[Testing the rollovers detection model](#testing)<br/>
[Training the rollovers detection model](#training)<br/>
[Using the rollovers detection model](#using)<br/>

<a name="preparing"/>

<H2 CLASS="western">Preparing the rollovers detection model:</H2>
The detection of rollovers is based on deep learning. You must first install tensorflow 1.15.0, tensorflow-hub 0.12.0 and keras 2.7.0 on your machine. It may be better to first create an anaconda environment for this purpose. ZZDeepRollover has been tested with Python 3.7 and may not work with other versions of Python (it won't work with Python 3.9 for example).<br/><br/>
You then need to place the output result folders of <a href="https://github.com/oliviermirat/ZebraZoom" target="_blank">ZebraZoom</a> inside the folder "ZZoutput" of this repository.<br/><br/>
In order to train the rollovers detection model, you must also manually classify the frames of some of the tracked videos in order to be able to create a training set. Look inside the folder "manualClassificationExamples" for examples of how to create such manual classifications. You then need to place those manual classifications inside the corresponding output result folders of ZebraZoom.<br/><br/>

<a name="testing"/>

<H2 CLASS="western">Testing the rollovers detection model:</H2>
In order to test the accuracy of the rollovers detection model, you can use the script leaveOneOutVideoTest.py, you will need to adjust some variables at the beginning of that script. The variable "videos" is an array that must contain the name of videos for which a manual classification of frames exist and has been placed inside the corresponding output result folder (inside the folder ZZoutput of this repository).<br/><br/>
The script leaveOneOutVideoTest.py will loop through all the videos learning the model on all but one video and testing on the video left out.<br/><br/>

<a name="training"/>

<H2 CLASS="western">Training the rollovers detection model:</H2>
Once the model has been tested using the steps described in the previous section, you can now learn the final model on all the videos for which a manual classification of frames exist using the script trainModel.py (you will need to adjust a few variables in that script).<br/><br/>

<a name="using"/>

<H2 CLASS="western">Using the rollovers detection model:</H2>
As mentionned above, you can then use the script detectRolloverFramesWithNewZZversion.py to apply the rollovers detection model on a video.<br/><br/>



