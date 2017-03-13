python train.py -c ./config/tt100k_classif.py -e TrafficSignVgg16_DataAugment -l ./data -s ./data
python train.py -c ./config/tt100k_classif_resnet50.py -e TrafficSignResNet50 -l ./data -s ./data
python train.py -c ./config/tt100k_classif_InceptionV3.py -e TrafficSignInceptionV3 -l ./data -s ./data
