#!/bin/csh

source /cs/casmip/bella_fadida/virtualenvs/keras_tensorflow/bin/activate.csh
module load tensorflow
#
# shell script running training
#

echo 'starting train script' | mutt -s 'starting train script' -- 

python3 train.py with /cs/casmip/bella_fadida/code/code_bella/config/config.json > log_train_1.txt
#echo 'finished running train 1' | mutt -a '/cs/casmip/bella_fadida/code/bella_code/log/log_train_1.txt' -s 'finished running train 1' -- bella.specktor@mail.huji.ac.il
