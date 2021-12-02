python3 train.py \
	--name lavalindoor \
	--dataset_mode lavalindoor \
	--dataroot /home/fangneng.zfn/datasets/LavalIndoor/tpami/ \
	--display_freq 1000 \
	--batchSize 16 \
	--niter 100 \
	--niter_decay 100 \
	--gpu_ids 0,1 \
	--continue_train