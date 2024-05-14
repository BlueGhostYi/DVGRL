"""
Created on March 1, 2023
PyTorch Implementation of DVGRL
Dual Variational Graph Reconstruction Learning for Social Recommendation
"""
__author__ = "Yi Zhang"

import Parser
import time
import torch
import torch.optim as optim
import numpy as np
import DVGRL
import batch_test
import data as d
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def main():
    args = Parser.parse_args()

    batch_test.set_seed(args.seed)
    loader = d.DataLoader(args.path + args.dataset + '/',args.sparsity)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    train_data = loader.train_data
    social_data = loader.social_data

    users_num = train_data.shape[0]
    items_num = train_data.shape[1]
  
    id_list = list(range(users_num))
    np.random.shuffle(id_list)
    
    p_dims = [args.dims, items_num]
    q_dims = [items_num, args.dims]

    print("=====================================================")
    print("Model          ", args.model)
    print("-----------------------------------------------------")
    print("Dataset        ", args.dataset)
    print("-----------------------------------------------------")
    print("User Num       ", users_num)
    print("-----------------------------------------------------")
    print("Item Num       ", items_num)
    print("-----------------------------------------------------")
    print("Train Num      ", loader.train_length)
    print("-----------------------------------------------------")
    print("Test Num       ", loader.test_length)
    print("-----------------------------------------------------")
    print("Social Num     ", loader.social_length)
    print("-----------------------------------------------------")
    print("Batch Size     ", args.batch_size)
    print("-----------------------------------------------------")
    print("LR             ", args.lr)
    print("-----------------------------------------------------")
    print("Model Structure", q_dims, p_dims)
    print("-----------------------------------------------------")
    print("beta           ", args.anneal_cap)
    print("-----------------------------------------------------")
    print("dropout        ", args.dropout)
    print("=====================================================")
    
    model = DVGRL.DVGRL(p_dims=p_dims, q_dims=q_dims, dropout=args.dropout, dataset=loader, device=device, args=args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    best_report_recall = [0., 0., 0.]
    best_report_ndcg = [0., 0., 0.]
    best_report_epoch = 0
    tolerate = 0

    if args.write:
        path_str = "./results/" + args.model + "_" + args.dataset + "_" + str(args.lr) + "_" + str(args.dims) + "_" + str(args.dropout) + ".txt"
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        a_recon_loss = 0.0
        s_recon_loss = 0.0

        start_time = time.time()
        count = 0
        for batch_id, start_id in enumerate(range(0, users_num, args.batch_size)):
            end_id = min(start_id + args.batch_size, users_num)
            batch_data = train_data[id_list[start_id:end_id]]
            batch_social_data = social_data[id_list[start_id:end_id]]
            
            users = torch.Tensor(id_list[start_id:end_id]).long()
            users = users.to(device)
            data = naive_sparse2tensor(batch_data).to(device)
            s_data = naive_sparse2tensor(batch_social_data).to(device)
            
            optimizer.zero_grad()

            recon_batch, recon_batch_s, mu, logvar, s_mu, s_logvar, u_z, s_z = model(users)

            recon_loss = model.recon_loss(recon_batch, data, mu, logvar, args.anneal_cap)
            social_loss = model.recon_loss(recon_batch_s, s_data, s_mu, s_logvar, args.anneal_cap)

            loss = recon_loss + social_loss

            loss.backward()
            train_loss += loss.item()

            a_recon_loss += recon_loss.item()
            s_recon_loss += social_loss.item()

            optimizer.step()
            count += 1
    
        average_loss = train_loss / count
        end_time = time.time()

        if args.show_loss:
            print("\t Epoch: %4d| train time: %.3f | train_loss:%.4f=%.4f + %.4f"
                  % (epoch + 1, end_time - start_time, average_loss, a_recon_loss/count, s_recon_loss/count))

        if epoch % args.log_interval == 0:
            if args.sparsity:
                result = batch_test.Test_sparsity(loader, model, device, [10, 20, 30], 0, args.batch_size)
                print("epoch:", epoch + 1, "==============================================")
                print("level_1: recall:", result[0]['recall'], ",precision:", result[0]['precision'], ',ndcg:', result[0]['ndcg'])
                print("level_2: recall:", result[1]['recall'], ",precision:", result[1]['precision'], ',ndcg:', result[1]['ndcg'])
                print("level_3: recall:", result[2]['recall'], ",precision:", result[2]['precision'], ',ndcg:', result[2]['ndcg'])
                print("level_4: recall:", result[3]['recall'], ",precision:", result[3]['precision'], ',ndcg:', result[3]['ndcg'])

            else:
                result = batch_test.Test(loader, model, device, [10, 20, 30], 0, args.batch_size)
                print("epoch:", epoch, " recall:", result['recall'], ",precision:", result['precision'], ',ndcg:',
                      result['ndcg'])

                if args.write:
                    final_perfs = "epoch:%.4d|recall:%.5f|precision:%.5f|ndcg:%.5f" % (epoch, result['recall'][1],result['precision'][1],result['ndcg'][1])
                    with open(path_str, "a") as f:
                        f.write(final_perfs + "\n")
                    f.close()

                if result['recall'][1] > best_report_recall[1]:
                    best_report_epoch = epoch + 1
                    best_report_recall = result['recall']
                    best_report_ndcg = result['ndcg']
                    tolerate = 0
                else:
                    tolerate += 1
                    if tolerate >= args.stop:
                        print("Early stopping at epoch ", epoch + 1)
                        print("best epoch:", best_report_epoch, "|best recall", best_report_recall, "|best ndcg:", best_report_ndcg)
                        exit()

            
if __name__ == '__main__':
    main()







