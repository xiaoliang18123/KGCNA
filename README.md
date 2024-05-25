# KGCNA: Knowledge Graph Collaborative Neighbor Awareness Network for Recommendation

This is our PyTorch implementation for the paper:

> Guangliang He , Zhen Zhang , Hanrui Wu , Sanchuan Luo, and Yudong Liu (2024). KGCNA: Knowledge Graph Collaborative Neighbor Awareness Network for Recommendation. in IEEE Transactions on Emerging Topics in Computational Intelligence.

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

- pytorch == 1.5.0
- numpy == 1.15.4
- scipy == 1.1.0
- sklearn == 0.20.0
- torch_scatter == 2.0.5
- networkx == 2.5

## Reproducibility & Example to Run the Codes

To demonstrate the reproducibility of the best performance reported in our paper and faciliate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (might be different for the custormized datasets) in the scripts.

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). 

- Last-fm dataset

```
nohup python main.py --dataset last-fm --batch_size 1024 --test_batch_size 4096 --test_sample_item_size 128 --train_sample_item_size 64 --user_neibor_size 64 --item_neibor_size 8  --context_hops 3 --gpu_id 0 --l2 1e-05  >last-fmu64i8_sp_test128_train64_l2_1e-05.txt 2>&1 &
```

- Yelp2018 dataset

```
nohup python main.py --dataset yelp2018 --batch_size 1024 --test_batch_size 4096 --test_sample_item_size 16 --train_sample_item_size 8 --user_neibor_size 32 --item_neibor_size 64  --context_hops 3 --gpu_id 0  --l2 1e-04 >yelp2018u32i64_train_sp8_test_sp16_l2_1e-04.txt 2>&1 &
```

- Alibaba-iFashion dataset

```
nohup python main.py --dataset alibaba-fashion --batch_size 1024 --test_batch_size 4096 --test_sample_item_size 8 --train_sample_item_size 4 --user_neibor_size 64 --item_neibor_size 64  --context_hops 3 --gpu_id 0 --l2 1e-04 >alibaba-fashionu64i64_train_sp4_test_sp8_lr_1e-04.txt 2>&1 &
```

