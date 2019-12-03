# DNLN model (x4)

# for lr_path in /home/xx/WangHua/数据集/RBPN_Vid4/bicubic_LR/*
# do
# hr_path=`echo $lr_path | sed -e "s/bicubic_LR/HR/g"`
# python main.py --save_results --test_only --resume -1 --data_test Demo --dir_demo $lr_path --dir_demo_GT $hr_path
# done
# echo "end！！！"

python main.py --save_results --chop