for alg in scaffold moon
do
	CUDA_VISIBLE_DEVICES=0 python experiments_moon_scaffold.py \
		--model=convnet \
		--dataset=cifar10 \
		--alg=$alg \
		--lr=0.01 \
		--batch-size=64 \
		--epochs=1 \
		--n_parties=10 \
		--rho=0.9 \
		--comm_round=2 \
		--partition=noniid-labeldir \
		--beta=0.5 \
		--device='cuda' \
		--datadir='./data/' \
		--logdir='./logs/' \
		--noise=0 \
		--init_seed=42
done