import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KGCA")

    # ===== dataset ===== #
    parser.add_argument('--fold', type=int, default=4, help='number of user popularity subset')
    parser.add_argument('--model_type', nargs='?', default='kgca',
                        help='Specify a loss type from {kgca}.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models and report performence.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument("--dataset", nargs="?", default="last-fm", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument('--test_sample_item_size', type=int, default=64,
                        help='fixed number of sample_item for each user')
    parser.add_argument('--train_sample_item_size', type=int, default=64,
                        help='fixed number of sample_item for each user')
    parser.add_argument('--user_neibor_size', type=int, default=8, help='neibor number of user')
    parser.add_argument('--item_neibor_size', type=int, default=8, help='neibor number of item')
    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")

    parser.add_argument("--batch_test_flag", type=bool, default=True, help="batch_test or not")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')

    parser.add_argument('--auc', nargs='?', type=bool, default=True, help='whether need auc indicator')
    parser.add_argument('--f1', nargs='?', type=bool, default=False, help='whether need f1 indicator or not')
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")
    # ===== gridsearch ===== #
    parser.add_argument("--gridsearch_user_neibor_size", nargs="?", default="[4, 8, 16, 32]",
                        help="Choose search user_neibor_list")
    parser.add_argument("--gridsearch_item_neibor_size", nargs="?", default="[4, 8, 16, 32]",
                        help="Choose search item_neibor_list")
    parser.add_argument("--gridsearch_user_sample_size", nargs="?", default="[64]",
                        help="Choose search sample_size_list")
    parser.add_argument("--gridsearch_context", nargs="?", default="[3]",
                        help="Choose search context_list")
    parser.add_argument("--gridsearch_lr", nargs="?", default="[1e-4, 1e-3, 1e-2]",
                        help="Choose search lr")
    parser.add_argument("--gridsearch_l2", nargs="?", default="[1e-4, 1e-3]",
                        help="Choose search l2")
    return parser.parse_args()
