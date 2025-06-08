for dataset in cifar100
do
	for beta in 0.1
	do
		for seed in 42
		do
			python experiments_moon.py \
				--model=convnet \
				--dataset=$dataset \
				--alg=moon \
				--lr=0.01 \
				--batch-size=64 \
				--epochs=10 \
				--mu=1 \
				--n_parties=10 \
				--rho=0.9 \
				--comm_round=20 \
				--partition=noniid-labeldir \
				--beta=$beta \
				--device='mps' \
				--datadir='./data/' \
				--logdir='./logs/' \
				--noise=0 \
				--init_seed=$seed
			sleep 5m
		done 
	done
done
