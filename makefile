train:
	python3 train.py --save_dir=./weights/mixed_ --log_file=./logs/mixed_logs.txt --dataset=mixed

predict:
	python3 predict.py --net=./weights/final_mixed_r50.pth

eval-wf:
	python3 extract_preds.py --net=./weights/final_mixed_r50.pth --save_dir=./data/mixed_wf/ --data=0
	python3 ./evaluate/wider_eval/evaluation.py -p ./data/mixed_wf/ -g ./evaluate/wider_eval/ground_truth/

eval-icf:
	python3 extract_preds.py --net=./weights/final_mixed_r50.pth --save_dir=./data/mixed_icf.csv --data=1
	python3 ./evaluate/eval_others/run.py ./data/mixed_icf.csv ./evaluate/eval_others/personai_icartoonface_detval.csv

eval-m109val:
	python3 extract_preds.py --net=./weights/final_mixed_r50.pth --save_dir=./data/mixed_m109val.csv --data=2
	python3 ./evaluate/eval_others/run.py ./data/mixed_m109val.csv ./evaluate/eval_others/manga109_val_annot.csv

eval-m109test:
	python3 extract_preds.py --net=./weights/final_mixed_r50.pth --save_dir=./data/mixed_m109test.csv --data=3
	python3 ./evaluate/eval_others/run.py ./data/mixed_m109test.csv ./evaluate/eval_others/manga109_test_annot.csv

clear:
	rm -r .ipynb_checkpoints -f
	rm -r */.ipynb_checkpoints -f
	rm -r */*/.ipynb_checkpoints -f
	rm -r __pycache__ -f
	rm -r */__pycache__ -f
	rm -r */*/__pycache__ -f
