<H1 CLASS="western" style="text-align:center;">ZZDeepRollover</H1>

This code enables the detection of rollovers performed by zebrafish larvae tracked by the open-source software <a href="https://github.com/oliviermirat/ZebraZoom" target="_blank">ZebraZoom</a>.<br/>
For more information visit <a href="https://zebrazoom.org/" target="_blank">zebrazoom.org</a> or email us at info@zebrazoom.org<br/>

<H2 CLASS="western">Table of content:</H2>

[Preparing the rollovers detection model](#preparing)<br/>
[Testing the rollovers detection model](#testing)<br/>
[Using the rollovers detection model](#using)<br/>

<a name="preparing"/>

<H2 CLASS="western">Preparing the rollovers detection model:</H2>
The detection of rollovers is based on deep learning. You must first install tensorflow and keras on your machine.<br/><br/>
You then need to place the output result folders of <a href="https://github.com/oliviermirat/ZebraZoom" target="_blank">ZebraZoom</a> inside the folder "ZZoutput" of this repository.<br/><br/>
In order to train the rollovers detection model, you must also manually classify the frames of some of the tracked videos in order to be able to create a training set. Look inside the folder "manualClassificationExamples" for examples of how to create such manual classifications. You then need to place those manual classifications inside the corresponding output result folders of ZebraZoom.<br/><br/>

<a name="testing"/>

<H2 CLASS="western">Testing the rollovers detection model:</H2>
In order to test the accuracy of the rollovers detection model, you can use the script leaveOneOutVideoTest.py, you will need to adjust some variables at the beginning of that script. The variable "videos" is an array that must contain the name of videos for which a manual classification of frames exist and has been placed inside the corresponding output result folder (inside the folder ZZoutput of this repository).<br/><br/>
The script leaveOneOutVideoTest.py will loop through all the videos learning the model on all but one video and testing on the video left out.<br/><br/>
Please note that ZZDeepRollover was tested on output folders generated using an older version of ZebraZoom (not the one that is open-sourced). We haven't tested with the newest version of ZebraZoom yet, so if you notice some issues please contact us at info@zebrazoom.org.<br/><br/>

<a name="using"/>

<H2 CLASS="western">Using the rollovers detection model:</H2>
Once the model has been tested using the steps described in the previous section, you can now learn the final model on all the videos for which a manual classification of frames exist using the script trainModel.sh (you will need to adjust a few variables in that script).<br/><br/>
Once the model has been learned, you can use the script generateLaunchScript.sh passing as an argument the path to a folder containing all the videos that you want to analyze (these videos must be in the format of a ZebraZoom output folder). The script generateLaunchScript.sh will generate a script called launch.sh. Launch this file launch.sh to apply the rollovers detection model on all the videos inside the folder that you previously selected.<br/><br/>

