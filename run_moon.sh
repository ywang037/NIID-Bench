for dataset in cifar10 cifar100 fmnist
do
	for beta in 0.02 0.05 0.1 0.2
	do
		CUDA_VISIBLE_DEVICES=2 python experiments_moon.py \
			--model=convnet \
			--dataset=$dataset \
			--alg=moon \
			--lr=0.01 \
			--batch-size=64 \
			--epochs=10 \
			--n_parties=10 \
			--rho=0.9 \
			--comm_round=2 \
			--partition=noniid-labeldir \
			--beta=$beta \
			--device='cuda' \
			--datadir='./data/' \
			--logdir='./logs/' \
			--noise=0 \
			--init_seed=42
	done
done
