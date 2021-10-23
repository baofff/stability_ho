import core.ablo as ablo
import core.random_search as rs
from core.loss import *
from core.utils import *
from core.datasets import *


def run(args):
    backup_args(args.__dict__, args.workspace_root)
    set_logger(os.path.join(args.workspace_root, 'log.txt'))

    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset == 'omniglot':
        data_gen_fn = Omniglot(args.width, flatten=True, num_classes=args.num_classes).get_data
    elif args.dataset == 'corrupted_mnist':
        data_gen_fn = CorruptedMnist(args.width, 'classification', flatten=True).get_data
    else:
        raise NotImplementedError

    if args.loss == 'MLPFeatureLearning':
        loss_cls = MLPFeatureLearning(lamb_mlp_shape=args.lamb_mlp_shape, theta_mlp_shape=args.theta_mlp_shape, task=args.task)
    elif args.loss == 'ReweightingMLP':
        loss_cls = ReweightingMLP(args.m_tr, args.theta_mlp_shape)
    else:
        raise NotImplementedError

    tr_dataset, val_dataset, te_dataset, mval_dataset = data_gen_fn(args.m_tr, args.m_val, args.m_te, args.m_mval, args.x_dim)
    tr_batch_size = len(tr_dataset) if 'batch_size' not in args else args.batch_size
    val_batch_size = len(val_dataset) if 'batch_size' not in args else args.batch_size
    te_batch_size = len(te_dataset) if 'batch_size' not in args else args.batch_size
    mval_batch_size = len(mval_dataset) if 'batch_size' not in args else args.batch_size

    if args.loss in ['ReweightingLinear', 'ReweightingMLP']:
        record = {"loss_val": [], "zero_one_loss_val": [],
                  "loss_te": [], "zero_one_loss_te": [],
                  "loss_mval": [], "zero_one_loss_mval": [], "gap": []}
    else:
        record = {"loss_tr_sgd": [], "loss_tr": [], "zero_one_loss_tr": [],
                  "loss_val": [], "zero_one_loss_val": [],
                  "loss_te": [], "zero_one_loss_te": [],
                  "loss_mval": [], "zero_one_loss_mval": [], "gap": []}
    if args.ho_algo == 'ablo':
        ablo.train(tr_dataset, val_dataset, te_dataset, mval_dataset, loss_cls, args.K, args.T, args.lr_l, args.wd_l, args.lr_h,
                   args.wd_h, args.mm_h, tr_batch_size, val_batch_size, te_batch_size, mval_batch_size, device, record)
    elif args.ho_algo == 'random_search':
        rs.train(tr_dataset, val_dataset, te_dataset, loss_cls, loss_cls.lamb_gen(args.T, requires_grad=False),
                 args.K, args.lr_l, args.wd_l, tr_batch_size, val_batch_size, te_batch_size, device, record)

    torch.save(record, os.path.join(args.workspace_root, "record.pt"))
    plot_record(record, args.workspace_root)
