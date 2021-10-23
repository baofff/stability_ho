from core.run import run
from core.task_schedule import wait_schedule, Task, available_devices
from multiprocessing import Process
import datetime
import argparse


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tag = 'ablo_K'

    if tag == 'ablo_K':
        tasks = []
        for seed in [33, 777, 1341, 25731]:
            for K in [1, 4, 16, 64, 128, 256]:
                args = argparse.Namespace(
                    T=10000, K=K, seed=seed, ho_algo='ablo', dataset='omniglot', num_classes=100, width=28, x_dim=784,
                    loss='MLPFeatureLearning', task='classification',
                    lamb_mlp_shape=[784, 256], theta_mlp_shape=[256, 128, 100], batch_size=50,
                    m_tr=500, m_val=100, m_te=1000, m_mval=100, lr_h=0.1, wd_h=0., mm_h=0, lr_l=0.1, wd_l=0.)
                args.workspace_root = "workspace/runs/fl_omniglot/ablo_K_{}/seed_{}_K_{}".format(now, seed, K)
                p = Process(target=run, args=(args,))
                tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'wdh':
        tasks = []
        for seed in [33, 777, 1341, 25731]:
            for K in [256]:
                for wd_h in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
                    args = argparse.Namespace(
                        T=10000, K=K, seed=seed, ho_algo='ablo', dataset='omniglot', num_classes=100, width=28, x_dim=784,
                        loss='MLPFeatureLearning', task='classification',
                        lamb_mlp_shape=[784, 256], theta_mlp_shape=[256, 128, 100], batch_size=50,
                        m_tr=500, m_val=100, m_te=1000, m_mval=100, lr_h=0.1, wd_h=wd_h, mm_h=0, lr_l=0.1, wd_l=0.)
                    args.workspace_root = "workspace/runs/fl_omniglot/wdh_{}/seed_{}_K_{}_wdh_{}".format(now, seed, K, wd_h)
                    p = Process(target=run, args=(args,))
                    tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'wdl':
        tasks = []
        for seed in [33, 777, 1341, 25731]:
            for K in [256]:
                for wd_l in [1e-2, 3e-2, 1e-1]:
                    args = argparse.Namespace(
                        T=10000, K=K, seed=seed, ho_algo='ablo', dataset='omniglot', num_classes=100, width=28, x_dim=784,
                        loss='MLPFeatureLearning', task='classification',
                        lamb_mlp_shape=[784, 256], theta_mlp_shape=[256, 128, 100], batch_size=50,
                        m_tr=500, m_val=100, m_te=1000, m_mval=100, lr_h=0.1, wd_h=0., mm_h=0, lr_l=0.1, wd_l=wd_l)
                    args.workspace_root = "workspace/runs/fl_omniglot/wdl_{}/seed_{}_K_{}_wdl_{}".format(now, seed, K, wd_l)
                    p = Process(target=run, args=(args,))
                    tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'wdboth':
        tasks = []
        for seed in [33, 777, 1341, 25731]:
            for K in [256]:
                for wd_l in [1e-2]:
                    for wd_h in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
                        args = argparse.Namespace(
                            T=10000, K=K, seed=seed, ho_algo='ablo', dataset='omniglot', num_classes=100, width=28, x_dim=784,
                            loss='MLPFeatureLearning', task='classification',
                            lamb_mlp_shape=[784, 256], theta_mlp_shape=[256, 128, 100], batch_size=50,
                            m_tr=500, m_val=100, m_te=1000, m_mval=100, lr_h=0.1, wd_h=wd_h, mm_h=0, lr_l=0.1, wd_l=wd_l)
                        args.workspace_root = "workspace/runs/fl_omniglot/wdboth_{}/seed_{}_K_{}_wdl_{}_wdh_{}".format(now, seed, K, wd_l, wd_h)
                        p = Process(target=run, args=(args,))
                        tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'rs_K':
        tasks = []
        for seed in [33, 777, 1341, 25731]:
            for K in [1, 4, 16, 64, 128, 256]:
                args = argparse.Namespace(
                    T=10000, K=K, seed=seed, ho_algo='random_search', dataset='omniglot', num_classes=100, width=28, x_dim=784,
                    loss='MLPFeatureLearning', task='classification',
                    lamb_mlp_shape=[784, 256], theta_mlp_shape=[256, 128, 100], batch_size=50,
                    m_tr=500, m_val=100, m_te=1000, m_mval=100, lr_l=0.1, wd_l=0.)
                args.workspace_root = "workspace/runs/fl_omniglot/rs_K_{}/seed_{}_K_{}".format(now, seed, K)
                p = Process(target=run, args=(args,))
                tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)
