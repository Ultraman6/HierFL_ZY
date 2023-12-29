# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from average import average_weights, models_are_equal


class Edge():

    def __init__(self, id, cids, shared_layers, share_dataloader=None):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}  # 暂存客户端上传的模型参数
        self.shared_state_dict = {}  # 全局模型参数
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.share_dataloader = share_dataloader

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client): # 注册用户信息同时决定是否把共享数据集传入
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        # 数据量还要加上共享数据集
        if self.share_dataloader != None:
            self.sample_registration[client.id] += len(self.share_dataloader.dataset)
            client.combine_share_data(self.share_dataloader)

    def receive_from_client(self, client_id, cshared_state_dict):
        self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        # print(models_are_equal(self.receiver_buffer[self.cids[0]], self.receiver_buffer[self.cids[1]]))
        sample_num = [snum for snum in self.sample_registration.values()]
        # d1 = copy.deepcopy(self.shared_state_dict)
        self.shared_state_dict = average_weights(w = received_dict,
                                                 s_num= sample_num)
        # print(models_are_equal(self.shared_state_dict, d1))

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        # print(models_are_equal(shared_state_dict, self.shared_state_dict))
        self.shared_state_dict = shared_state_dict
        return None



