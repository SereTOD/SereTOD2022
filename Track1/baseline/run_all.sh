cd entity-extraction 
bash run.sh 6
cd ../entity-coreference
bash run.sh 6
cd ../slot-filling
bash run.sh 6
cd ../entity-slot-alignment
bash run.sh 6
cd .. 
python get_submissions.py
python eval_script2.py