# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:25:26 2022

@author: lixyxd
"""

import argparse


def set_opts():
    parser = argparse.ArgumentParser()
    # trainning settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batchsize of training, (default:64)")
    parser.add_argument('--test_batchsize', type=int, default=8,
                        help="Batchsize of training, (default:64)")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs,  (default:60)")
    parser.add_argument('--lr', type=float, default=2e-3,  # 1e-3
                        help="Initialized learning rate, (default: 1e-5)")

    parser.add_argument('--gamma', type=float, default=0.9,
                        help="Decaying rate for the learning rate, (default: 0.3)")
    parser.add_argument('-p', '--print_freq', type=int, default=100,
                        help="Print frequency (default: 100)")
    parser.add_argument('-s', '--save_model_freq', type=int, default=5,
                        help="Save model frequency (default: 1)")
    parser.add_argument('--csf', type=int, default=20,
                        help="Save canshu frequency (default: 1)")

    # GPU settings
    parser.add_argument('--gpu_id', type=int, nargs='+', default=0,
                        help="GPU ID, which allow multiple GPUs")

    # dataset settings
    parser.add_argument('--Train_dir', default="./mixdata/mix_data_76800.mat", type=str, metavar='PATH',
                        help="Path to save the SIDD dataset, (default: './data/data_train_mix4_25600-0.mat')")
    parser.add_argument('--Test_dir', default='./mixdata/data2_test_92160.mat', type=str,
                        metavar='PATH', help="Path to save the images, (test_data_92160default: ./完整混合测试集/data_test25600_20dB.mat)")

    parser.add_argument('--labelTrain_dir', default='./mixdata/mix_label_76800.mat', type=str,
                        metavar='PATH', help="Path to save the SIDD dataset, (default: ./data/label_train_p3_25600.mat)")
    parser.add_argument('--labelTest_dir', default='./mixdata/label2_test_92160.mat', type=str,
                        metavar='PATH', help="Path to save the images, (test_label_92160default: ./完整混合测试集/label_test_25600.mat)")

    parser.add_argument('--Nr0_dir', default='./data/Nr0_38.mat', type=str, metavar='PATH',
                        help="Path to save the SIDD dataset, (default: None)")
    parser.add_argument('--Na0_dir', default='./data/Na01.mat', type=str,
                        metavar='PATH', help="Path to save the images, (default: None)")
    parser.add_argument('--G_dir', default='./data/R256_21.npy', type=str,
                        metavar='PATH', help="Path to save the images, (default: None)")
    parser.add_argument('--GG_dir', default='./data/RR256_21.npy', type=str,
                        metavar='PATH', help="Path to save the images, (default: None)")

    parser.add_argument('--Nr', type=int, default=16384, help="The depth of SNet, (default: 4)")
    parser.add_argument('--Na', type=int, default=1, help="The depth of SNet, (default: 4)")
    parser.add_argument('--N', type=int, default=256, help="The depth of SNet, (default: 4)")
    parser.add_argument('--sparse_Band', type=int, default=0, help="The depth of SNet, (default: 4)")
    parser.add_argument('--sparse_Azimuth', type=int, default=1, help="The depth of SNet, (default: 4)")

    # model and log saving
    parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH',
                        help="Path to save the log file, (default: ./log)")
    parser.add_argument('--model_dir', default='./model_cnn00', type=str, metavar='PATH',
                        help="Path to save the model file, (default: ./model)")
    parser.add_argument('--num_workers', default=8, type=int,
                        help="Number of workers to load data, (default: 8)")
    # network architecture
    parser.add_argument('--SNR_min', type=float, default=5,
                        help="Initial value for LeakyReLU, (default: 0.2)")
    parser.add_argument('--SNR_max', type=float, default=5,
                        help="Initial value for LeakyReLU, (default: 0.2)")
    parser.add_argument('--test_SNR', type=float, default=5,
                        help="Initial value for LeakyReLU, (default: 0.2)")
    # parser.add_argument('--K', type=int, default=8, help="The iteration of Module, (default: 4)")
    parser.add_argument('--layer_num', type=int, default=20, help="The depth of SNet, (default: 20)")

    args = parser.parse_args()

    return args
