for alg in scaffold
do
	python experiments.py \
		--model=simple-cnn \
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
		--device='cpu' \
		--datadir='./data/' \
		--logdir='./logs/' \
		--noise=0 \
		--init_seed=0
done