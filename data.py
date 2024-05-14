import scipy.sparse as sp
import numpy as np

class DataLoader():
    def __init__(self, path, sparsity):
        self.path = path
        self.num_users = 0
        self.num_items = 0

        users_list, items_list = self.load_data('train.txt')
        test_users_list, test_items_list = self.load_data('test.txt')
        follower_list, followee_list = self.load_social_data("social.txt")
        
        self.num_users += 1
        self.num_items += 1
        
        self.user_list = set(users_list)
        
        self.train_length = len(users_list)
        self.test_length = len(test_users_list)
        self.social_length = len(follower_list)
        
        self.num_nodes = self.num_users + self.num_items
        
        self.train_data = sp.csr_matrix((np.ones_like(users_list), (users_list, items_list)), 
                                              dtype='float64', shape=(self.num_users, self.num_items))
        self.social_data = sp.csr_matrix((np.ones_like(follower_list), (follower_list, followee_list)), 
                                              dtype='float64', shape=(self.num_users, self.num_users))
        
        self.bipartite_graph = self.sparse_adjacency_matrix()
        
        self.social_graph = self.sparse_mean_social_matrix()
        
        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))
        
        self.test_dict = self.build_test(test_users_list, test_items_list)

        if sparsity:
            self.split_test_dict, self.split_state = self.create_sparsity_split()
        
        
    def load_data(self, name):
        users_list = []
        items_list = []

        with open(self.path + name, 'r') as f:
            strs = f.readline()
            while strs is not None and strs != "":
                arr = strs.strip().split(" ")
                user_id = int(arr[0])
                items = list(arr[1:])
                pos_id = [int(i) for i in items]
                inter_num = len(items)
                
                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))
    
                users_list.extend([int(arr[0])] * inter_num)
                for i in range(len(items)):
                    items_list.append(int(items[i]))
                strs = f.readline()

        return users_list, items_list
       
    def load_social_data(self, name):
        followers_list = []
        followees_list = []
        with open(self.path + name, 'r') as f:
            strs = f.readline()
            while strs is not None and strs != "":
                arr = strs.strip().split(" ")
                follower_id = int(arr[0])
                followees = list(arr[1:])
                pos_id = [int(i) for i in followees]
                inter_num = len(followees)

                self.num_users = max(self.num_users, follower_id)
                self.num_users = max(self.num_users, max(pos_id))
    
                followers_list.extend([int(arr[0])] * (inter_num))
                for i in range(len(followees)):
                    followees_list.append(int(followees[i]))
                strs = f.readline()
        return followers_list, followees_list 
    
    def sparse_adjacency_matrix(self):
        try:
            pre_adjacency = sp.load_npz(self.path + '/pre_R_mat.npz')
            print("\t Adjacency matrix loading completed.")
            norm_R = pre_adjacency
        except:
            R = self.train_data.todok()
            
            row_sum = np.array(R.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = d_mat.dot(R)
            
            col_sum = np.array(R.sum(axis=0))
            d_inv = np.power(col_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = norm_R.dot(d_mat)

            norm_R = norm_R.tocsr()
            sp.save_npz(self.path + '/pre_R_mat.npz', norm_R)
            print("\t Adjacency matrix constructed.")

        self.bipartite_graph = norm_R

        return self.bipartite_graph
       
    def sparse_social_matrix(self):
        try:
            pre_adjacency = sp.load_npz(self.path + '/pre_S_mat.npz')
            print("\t Adjacency matrix loading completed.")
            norm_R = pre_adjacency
        except:
            R = self.social_data.todok()
            
            row_sum = np.array(R.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = d_mat.dot(R)
            
            col_sum = np.array(R.sum(axis=0))
            d_inv = np.power(col_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = norm_R.dot(d_mat)

            norm_R = norm_R.tocsr()
            sp.save_npz(self.path + '/pre_S_mat.npz', norm_R)
            print("\t Adjacency matrix constructed.")

        self.social_graph = norm_R

        return self.social_graph
       
    def sparse_mean_social_matrix(self):
        try:
            pre_adjacency = sp.load_npz(self.path + '/pre_S_mat.npz')
            print("\t Adjacency matrix loading completed.")
            norm_R = pre_adjacency
        except:
            R = self.social_data.todok()
            
            row_sum = np.array(R.sum(axis=1))
            d_inv = np.power(row_sum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = R.dot(d_mat)

            norm_R = norm_R.tocsr()
            sp.save_npz(self.path + '/pre_S_mat.npz', norm_R)
            print("\t Adjacency matrix constructed.")

        self.social_graph = norm_R

        return self.social_graph
    
    def get_user_pos_items(self, users):
        positive_items = []
        for user in users:
            positive_items.append(self.train_data[user].nonzero()[1])
        return positive_items
             
    def sample_data_to_train_all(self):
        users = np.random.randint(0, self.num_users, self.train_length)
        sample_list = []
        for i, user in enumerate(users):
            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue
            positive_index = np.random.randint(0, len(positive_items))
            positive_item = positive_items[positive_index]
            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    continue
                else:
                    break
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)
    
    def build_test(self, test_users_list, test_items_list):
        test_data = {}
        for i, item in enumerate(test_items_list):
            user = test_users_list[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = dict()

        for uid in all_users:
            train_iids = self.all_positive[uid]
            test_iids = self.test_dict[uid]

            num_iids = len(train_iids) + len(test_iids)

            if num_iids not in user_n_iid.keys():
                user_n_iid[num_iids] = [uid]
            else:
                user_n_iid[num_iids].append(uid)

        split_uids = list()
        temp = []
        count = 1
        fold = 4
        n_count = self.train_length + self.test_length
        n_rates = 0
        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.train_length + self.test_length):
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
