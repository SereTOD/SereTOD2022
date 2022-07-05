cd entity-extraction 
bash run.sh 1
cd ../entity-coreference
bash run.sh 1
cd ../slot-filling
bash run.sh 1
cd ../entity-slot-alignment
bash run.sh 1
cd .. 
python eval_script.py 