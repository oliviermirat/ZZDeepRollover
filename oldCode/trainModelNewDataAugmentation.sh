rm -r initialImages/*

recenterImageWindow=0
numberOfRotationsDataAugmentationTraining=1
epochsNbTraining=10

./cleanDataset.sh trainingDataset

videos=( '20190427-1-2-1' '20190427-2-2-8' '20190503-2-2-3' '20190727-3-2-2' '20190427-1-2-5' '20190503-1-2-5' '20190503-4-2-7' '20190727-5-2-7' '20190427-2-2-6' '20190503-2-2-1' '20190727-1-2-3' '20190727-6-2-4' )

for name in "${videos[@]}"
do
  python createInitialImages.py $name rolloverManualClassification.json ZZoutput/
  python createTrainOrTestDataset.py initialImages/ $name trainingDataset $numberOfRotationsDataAugmentationTraining $recenterImageWindow 0
done

echo "data preparation done"

./cleanModel.sh model
python learnModel.py trainingDataset $epochsNbTraining model 1
