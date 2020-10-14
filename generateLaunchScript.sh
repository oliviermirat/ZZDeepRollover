cd $1
rm launch.sh
for entry in *
do
  printf "python detectRolloverFrames.py trainingDataset "$entry" ZZoutput/ 5 24 0 1\n"  >> ../launch.sh
done
cd ..
