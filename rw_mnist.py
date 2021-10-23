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
        for seed in [0, 33, 777, 1341, 25731]:
            for K in [64, 128, 256, 512]:
                args = argparse.Namespace(
                    T=5000, K=K, seed=seed, ho_algo='ablo', dataset='corrupted_mnist', width=28, x_dim=784,
                    loss='ReweightingMLP', theta_mlp_shape=[784, 256, 10],
                    m_tr=2000, m_val=200, m_te=1000, m_mval=200, batch_size=100, lr_h=10, lr_l=0.3, wd_h=0., wd_l=0.,
                    mm_h=0.)
                args.workspace_root = "workspace/runs/rw_mnist/ablo_K_{}/seed_{}_K_{}".format(
                    now, seed, K)
                p = Process(target=run, args=(args,))
                tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'wdh':
        tasks = []
        for seed in [0, 33, 777, 1341, 25731]:
            for K in [512]:
                for wd_h in [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]:
                    args = argparse.Namespace(
                        T=5000, K=K, seed=seed, ho_algo='ablo', dataset='corrupted_mnist', width=28, x_dim=784,
                        loss='ReweightingMLP', theta_mlp_shape=[784, 256, 10],
                        m_tr=2000, m_val=200, m_te=1000, m_mval=200, batch_size=100, lr_h=10, lr_l=0.3, wd_h=wd_h, wd_l=0.,
                        mm_h=0.)
                    args.workspace_root = "workspace/runs/rw_mnist/wdh_{}/seed_{}_K_{}_wdh_{}".format(
                        now, seed, K, wd_h)
                    p = Process(target=run, args=(args,))
                    tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'wdl':
        tasks = []
        for seed in [0, 33, 777, 1341, 25731]:
            for K in [512]:
                for wd_l in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
                    args = argparse.Namespace(
                        T=5000, K=K, seed=seed, ho_algo='ablo', dataset='corrupted_mnist', width=28, x_dim=784,
                        loss='ReweightingMLP', theta_mlp_shape=[784, 256, 10],
                        m_tr=2000, m_val=200, m_te=1000, m_mval=200, batch_size=100, lr_h=10, lr_l=0.3, wd_h=0., wd_l=wd_l,
                        mm_h=0.)
                    args.workspace_root = "workspace/runs/rw_mnist/wdl_{}/seed_{}_K_{}_wdl_{}".format(
                        now, seed, K, wd_l)
                    p = Process(target=run, args=(args,))
                    tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'wdboth':
        tasks = []
        for seed in [0, 33, 777, 1341, 25731]:
            for K in [512]:
                for wd_l, wd_h in [(1e-3, 3e-6), (1e-3, 1e-5),
                                   (1e-3, 3e-5), (1e-3, 1e-4)]:
                        args = argparse.Namespace(
                            T=5000, K=K, seed=seed, ho_algo='ablo', dataset='corrupted_mnist', width=28, x_dim=784,
                            loss='ReweightingMLP', theta_mlp_shape=[784, 256, 10],
                            m_tr=2000, m_val=200, m_te=1000, m_mval=200, batch_size=100, lr_h=10, lr_l=0.3, wd_h=wd_h, wd_l=wd_l,
                            mm_h=0.)
                        args.workspace_root = "workspace/runs/rw_mnist/wdboth_{}/seed_{}_K_{}_wdl_{}_wdh_{}".format(now, seed, K, wd_l, wd_h)
                        p = Process(target=run, args=(args,))
                        tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)

    elif tag == 'rs_K':
        tasks = []
        for seed in [0, 33, 777, 1341, 25731]:
            for K in [64, 128, 256, 512]:
                args = argparse.Namespace(
                    T=5000, K=K, seed=seed, ho_algo='random_search', dataset='corrupted_mnist', width=28, x_dim=784,
                    loss='ReweightingMLP', theta_mlp_shape=[784, 256, 10],
                    m_tr=2000, m_val=200, m_te=1000, m_mval=200, batch_size=100, lr_l=0.3, wd_l=0.)
                args.workspace_root = "workspace/runs/rw_mnist/rs_K_{}/seed_{}_K_{}".format(
                    now, seed, K)
                p = Process(target=run, args=(args,))
                tasks.append(Task(p, 1))
        wait_schedule(tasks, devices=available_devices() * 3)
