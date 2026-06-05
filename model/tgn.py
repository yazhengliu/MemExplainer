import logging
import time

import numpy as np
import torch
from collections import defaultdict
# from .edge_relevance_explainer import EdgeRelevanceExplainer

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator, LastMessageAggregator, MeanMessageAggregator
from modules.message_function import get_message_function, IdentityMessageFunction, MLPMessageFunction
from modules.memory_updater import get_memory_updater, GRUMemoryUpdater, RNNMemoryUpdater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False,forbidden_memory_update=False):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst
    self.forbidden_memory_update = forbidden_memory_update

    self.node_raw_embed = (
        self.node_raw_features
    )  # just a copy for compatiblility in PGExplainerExt._create_explainer_input()
    self.edge_raw_embed = self.edge_raw_features



    if self.use_memory:
      self.memory_dimension = memory_dimension
      print('self.memory_dimension',self.memory_dimension)
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)


      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    print('self.node_raw_features',self.node_raw_features.shape)
    print('self.edge_raw_features', self.edge_raw_features.shape)

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)


    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)



  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, message_dict,memory_dict,message_trace_dict,n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    # print('self.use_memory',self.use_memory)
    #
    # print('self.memory_update_at_start',self.memory_update_at_start)

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches

        # print('self.memory.messages',type(self.memory.messages))
        memory, last_update,C_message,C_memory,C_message_trace = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages,message_dict,memory_dict,message_trace_dict)
        print('memory',memory.shape)
        print('last_update',last_update.shape)






      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update
      # if len(C_message)>0:
      #     print('C_message 0',C_message[1])

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module

    # node_embedding = self.embedding_module.compute_embedding(memory=memory,
    #                                                          source_nodes=nodes,
    #                                                          timestamps=timestamps,
    #                                                          n_layers=self.n_layers,
    #                                                          n_neighbors=n_neighbors,
    #                                                          time_diffs=time_diffs)
    if self.embedding_module_type=='graph_sum':
        node_embedding_iter, C_memory_features, \
            C_neighbor_memory_features, \
            temporal_edge_contributions, sample_neighbors, sample_neighbor_edgeidx = self.embedding_module.compute_embedding_iterative(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
        )
    elif self.embedding_module_type=='graph_attention':
        print('graph_attention')
        node_embedding_iter, C_memory_features, \
            C_neighbor_memory_features, \
            temporal_edge_contributions, sample_neighbors, sample_neighbor_edgeidx = self.embedding_module.compute_embedding_attention(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
        )




    # print(C_raw_features.shape)
    print('C_memory_features.shape',C_memory_features.shape)
    # print('C_neighbor_raw_features',C_neighbor_raw_features.shape)
    # print(C_source_time.shape)
    # print(C_neighbor_embeddings.shape)
    # print(C_edge_time_embeddings.shape)
    # print(C_edge_features.shape)

    print('node_embedding_iter',node_embedding_iter.shape)


    # total_contrib=C_raw_features.sum(dim=1)+C_memory_features.sum(dim=1)+C_source_time.sum(dim=1)+ \
    #               C_neighbor_raw_features.sum(dim=(1,2))+C_neighbor_memory_features.sum(dim=(1,2))+C_edge_time_embeddings.sum(dim=(1,2))\
    #               +C_edge_features.sum(dim=(1,2))

    # print('verify flag', torch.allclose(total_contrib, node_embedding_iter, atol=1e-4))

    node_embedding=node_embedding_iter

    # C_message_to_memory_features=dict()
    #
    # C_old_memory_to_memory_features = dict()
    #
    # C_memory_features = C_memory_features.to(torch.float64)
    #
    # for i in range(len(nodes)):
    #     if nodes[i] in C_message.keys():
    #         if nodes[i] not in C_message_to_memory_features:
    #             v = C_message[nodes[i]].to(dtype=C_memory_features.dtype, device=C_memory_features.device)
    #             # print('v.shape',v.shape)
    #             # print('C_memory_features[i]',C_memory_features[i].shape)
    #             C_message_to_memory_features[nodes[i]]=v@C_memory_features[nodes[i]]
    #
    #     if nodes[i] in C_memory.keys():
    #         if nodes[i] not in C_old_memory_to_memory_features:
    #             u = C_memory[nodes[i]].to(dtype=C_memory_features.dtype, device=C_memory_features.device)
    #             C_old_memory_to_memory_features[nodes[i]] = u @ C_memory_features[nodes[i]]
    #
    #     else:
    #         pass
    #         # print('yes',update_node)
    #
    # memory1_verify=True
    #
    # for i in range(len(nodes)):
    #     if nodes[i] in C_message_to_memory_features:
    #         test1=C_message_to_memory_features[nodes[i]].sum(dim=0)+C_old_memory_to_memory_features[nodes[i]].sum(dim=0)
    #         test2=C_memory_features[nodes[i]].sum(dim=0)
    #         if torch.allclose(test1, test2, atol=1e-4)==False:
    #             memory1_verify=False
    #         # print('memory1 verify', )
    #         # print('test1',)
    #         # print('test2',)
    # print('memory1_verify',memory1_verify)









    # if torch.allclose(node_embedding, node_embedding_iter, atol=1e-6, rtol=1e-5):
    #     print("两个 embedding 在数值上近似相等")
    # else:
    #     print(" 两个 embedding 存在差异")

    # print('node_embedding_iter',node_embedding_iter)
    # print('node_embedding', node_embedding)
    #print('edge_contributions',edge_contributions)


    # print('per_message_contrib',per_message_contrib)
    # print('per_old_memory_contrib',per_old_memory_contrib)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]




    if self.use_memory and  ( not self.forbidden_memory_update):
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        print('positives',positives.shape)
        # for key,value in self.memory.messages.items():
        #     if len(value)!=0:
        #         print('key', key)
        #         print('value', len(value))
        #         print(value)


        #print('memory.messages',self.memory.messages)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      # print('unique_sources',unique_sources)
      # print('source_id_to_messages',source_id_to_messages)
      # print('unique_destinations',unique_destinations)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    #   n_samples = len(source_nodes)
    #   total_contrib_source = total_contrib[:n_samples]
    #   total_contrib_destination = total_contrib[n_samples:2 * n_samples]
    #   total_contrib_negative = total_contrib[2 * n_samples:]
    #
    #   # 验证source部分
    #   source_check = torch.allclose(total_contrib_source, source_node_embedding, atol=1e-4)
    #   print(f"Source贡献值验证: {'通过' if source_check else '失败'}")
    #   if not source_check:
    #       print(f"Source差异: {torch.abs(total_contrib_source - source_node_embedding).max():.6f}")
    #
    #   # 验证destination部分
    #   dest_check = torch.allclose(total_contrib_destination, destination_node_embedding, atol=1e-4)
    #   print(f"Destination贡献值验证: {'通过' if dest_check else '失败'}")
    #   if not dest_check:
    #       print(f"Destination差异: {torch.abs(total_contrib_destination - destination_node_embedding).max():.6f}")
    #
    #   # 验证negative部分
    #   neg_check = torch.allclose(total_contrib_negative, negative_node_embedding, atol=1e-4)
    #   print(f"Negative贡献值验证: {'通过' if neg_check else '失败'}")
    #   if not neg_check:
    #       print(f"Negative差异: {torch.abs(total_contrib_negative - negative_node_embedding).max():.6f}")
    #
    #   # 总体验证
    #   overall_check = source_check and dest_check and neg_check
    #   print(f"总体贡献值验证: {'通过' if overall_check else '失败'}")
    #
    # print('source_node_embedding',source_node_embedding.shape)
    # print('destination_node_embedding', destination_node_embedding.shape)
    # print('negative_node_embedding',negative_node_embedding.shape)


    return source_node_embedding, destination_node_embedding, negative_node_embedding, C_memory_features,  \
               C_neighbor_memory_features, temporal_edge_contributions,C_message,sample_neighbors,sample_neighbor_edgeidx


  def compute_temporal_embeddings_withtime(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, message_dict,memory_dict,message_trace_dict,n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    # print('self.use_memory',self.use_memory)
    #
    # print('self.memory_update_at_start',self.memory_update_at_start)
    memory_time=0
    neighbor_time=0

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches

        # print('self.memory.messages',type(self.memory.messages))
        memory_start=time.time()
        memory, last_update,C_message,C_memory,C_message_trace = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages,message_dict,memory_dict,message_trace_dict)
        print('memory',memory.shape)
        print('last_update',last_update.shape)
        memory_end=time.time()

        memory_time=memory_end-memory_start






      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update
      # if len(C_message)>0:
      #     print('C_message 0',C_message[1])

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module

    # node_embedding = self.embedding_module.compute_embedding(memory=memory,
    #                                                          source_nodes=nodes,
    #                                                          timestamps=timestamps,
    #                                                          n_layers=self.n_layers,
    #                                                          n_neighbors=n_neighbors,
    #                                                          time_diffs=time_diffs)
    neighbor_start=time.time()

    node_embedding_iter, C_memory_features,  \
               C_neighbor_memory_features, \
              temporal_edge_contributions,sample_neighbors,sample_neighbor_edgeidx= self.embedding_module.compute_embedding_iterative(
                   memory=memory,
                   source_nodes=nodes,
                   timestamps=timestamps,
                   n_layers=self.n_layers,
                   n_neighbors=n_neighbors,
               )
    neighbor_end = time.time()

    neighbor_time=neighbor_end-neighbor_start

    # print(C_raw_features.shape)
    print('C_memory_features.shape',C_memory_features.shape)
    # print('C_neighbor_raw_features',C_neighbor_raw_features.shape)
    # print(C_source_time.shape)
    # print(C_neighbor_embeddings.shape)
    # print(C_edge_time_embeddings.shape)
    # print(C_edge_features.shape)

    print('node_embedding_iter',node_embedding_iter.shape)


    # total_contrib=C_raw_features.sum(dim=1)+C_memory_features.sum(dim=1)+C_source_time.sum(dim=1)+ \
    #               C_neighbor_raw_features.sum(dim=(1,2))+C_neighbor_memory_features.sum(dim=(1,2))+C_edge_time_embeddings.sum(dim=(1,2))\
    #               +C_edge_features.sum(dim=(1,2))

    # print('verify flag', torch.allclose(total_contrib, node_embedding_iter, atol=1e-4))

    node_embedding=node_embedding_iter

    # C_message_to_memory_features=dict()
    #
    # C_old_memory_to_memory_features = dict()
    #
    # C_memory_features = C_memory_features.to(torch.float64)
    #
    # for i in range(len(nodes)):
    #     if nodes[i] in C_message.keys():
    #         if nodes[i] not in C_message_to_memory_features:
    #             v = C_message[nodes[i]].to(dtype=C_memory_features.dtype, device=C_memory_features.device)
    #             # print('v.shape',v.shape)
    #             # print('C_memory_features[i]',C_memory_features[i].shape)
    #             C_message_to_memory_features[nodes[i]]=v@C_memory_features[nodes[i]]
    #
    #     if nodes[i] in C_memory.keys():
    #         if nodes[i] not in C_old_memory_to_memory_features:
    #             u = C_memory[nodes[i]].to(dtype=C_memory_features.dtype, device=C_memory_features.device)
    #             C_old_memory_to_memory_features[nodes[i]] = u @ C_memory_features[nodes[i]]
    #
    #     else:
    #         pass
    #         # print('yes',update_node)
    #
    # memory1_verify=True
    #
    # for i in range(len(nodes)):
    #     if nodes[i] in C_message_to_memory_features:
    #         test1=C_message_to_memory_features[nodes[i]].sum(dim=0)+C_old_memory_to_memory_features[nodes[i]].sum(dim=0)
    #         test2=C_memory_features[nodes[i]].sum(dim=0)
    #         if torch.allclose(test1, test2, atol=1e-4)==False:
    #             memory1_verify=False
    #         # print('memory1 verify', )
    #         # print('test1',)
    #         # print('test2',)
    # print('memory1_verify',memory1_verify)









    # if torch.allclose(node_embedding, node_embedding_iter, atol=1e-6, rtol=1e-5):
    #     print("两个 embedding 在数值上近似相等")
    # else:
    #     print(" 两个 embedding 存在差异")

    # print('node_embedding_iter',node_embedding_iter)
    # print('node_embedding', node_embedding)
    #print('edge_contributions',edge_contributions)


    # print('per_message_contrib',per_message_contrib)
    # print('per_old_memory_contrib',per_old_memory_contrib)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]




    if self.use_memory and  ( not self.forbidden_memory_update):
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        print('positives',positives.shape)
        # for key,value in self.memory.messages.items():
        #     if len(value)!=0:
        #         print('key', key)
        #         print('value', len(value))
        #         print(value)


        #print('memory.messages',self.memory.messages)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      # print('unique_sources',unique_sources)
      # print('source_id_to_messages',source_id_to_messages)
      # print('unique_destinations',unique_destinations)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    #   n_samples = len(source_nodes)
    #   total_contrib_source = total_contrib[:n_samples]
    #   total_contrib_destination = total_contrib[n_samples:2 * n_samples]
    #   total_contrib_negative = total_contrib[2 * n_samples:]
    #
    #   # 验证source部分
    #   source_check = torch.allclose(total_contrib_source, source_node_embedding, atol=1e-4)
    #   print(f"Source贡献值验证: {'通过' if source_check else '失败'}")
    #   if not source_check:
    #       print(f"Source差异: {torch.abs(total_contrib_source - source_node_embedding).max():.6f}")
    #
    #   # 验证destination部分
    #   dest_check = torch.allclose(total_contrib_destination, destination_node_embedding, atol=1e-4)
    #   print(f"Destination贡献值验证: {'通过' if dest_check else '失败'}")
    #   if not dest_check:
    #       print(f"Destination差异: {torch.abs(total_contrib_destination - destination_node_embedding).max():.6f}")
    #
    #   # 验证negative部分
    #   neg_check = torch.allclose(total_contrib_negative, negative_node_embedding, atol=1e-4)
    #   print(f"Negative贡献值验证: {'通过' if neg_check else '失败'}")
    #   if not neg_check:
    #       print(f"Negative差异: {torch.abs(total_contrib_negative - negative_node_embedding).max():.6f}")
    #
    #   # 总体验证
    #   overall_check = source_check and dest_check and neg_check
    #   print(f"总体贡献值验证: {'通过' if overall_check else '失败'}")
    #
    # print('source_node_embedding',source_node_embedding.shape)
    # print('destination_node_embedding', destination_node_embedding.shape)
    # print('negative_node_embedding',negative_node_embedding.shape)


    return source_node_embedding, destination_node_embedding, negative_node_embedding, C_memory_features,  \
               C_neighbor_memory_features, temporal_edge_contributions,C_message,sample_neighbors,sample_neighbor_edgeidx,memory_time,neighbor_time

  def compute_temporal_embeddings_without_contributions(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs,n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    # print('self.use_memory',self.use_memory)
    #
    # print('self.memory_update_at_start',self.memory_update_at_start)

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches

        # print('yes')
        memory, last_update = self.get_updated_memory_without_contribution(list(range(self.n_nodes)),
                                                      self.memory.messages)



      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update


      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    if self.embedding_module_type == 'graph_sum':
        node_embedding_iter = self.embedding_module.compute_embedding_iterative_without_contribution(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
        )

    elif self.embedding_module_type=='graph_attention':
        print('graph_attention')
        node_embedding_iter= self.embedding_module.compute_embedding_attention_without_contribution(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
        )

    node_embedding = node_embedding_iter






    # node_embedding = self.embedding_module.compute_embedding_original(memory=memory,
    #                                                          source_nodes=nodes,
    #                                                          timestamps=timestamps,
    #                                                          n_layers=self.n_layers,
    #                                                          n_neighbors=n_neighbors,
    #                                                          time_diffs=time_diffs)



    # print('node_embedding_iter',node_embedding_iter.shape)

    # print('node_embedding',node_embedding)








    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]




    if self.use_memory and  (not self.forbidden_memory_update):
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        # print('positives',positives.shape)
        # for key,value in self.memory.messages.items():
        #     if len(value)!=0:
        #         print('key', key)
        #         print('value', len(value))
        #         print(value)


        #print('memory.messages',self.memory.messages)
        self.update_memory(positives, self.memory.messages)

        assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      # print('unique_sources',unique_sources)
      # print('source_id_to_messages',source_id_to_messages)
      # print('unique_destinations',unique_destinations)
      # print('self.memory_update_at_start',self.memory_update_at_start)
      if self.memory_update_at_start:
        # print('start')
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        # print('end')
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]




    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings_without_contributions(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    source_node_embedding = source_node_embedding.float()
    destination_node_embedding = destination_node_embedding.float()
    negative_node_embedding = negative_node_embedding.float()

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps,_ = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages,message_dict,memory_dict,message_trace_dict):
    # Aggregate messages for the same nodes
    # print('len nodes',len(nodes))
    unique_nodes, unique_messages, unique_timestamps,edge_info_list = \
      self.message_aggregator.aggregate(
        nodes,
        messages) # last or mean
    # print('unique_nodes',len(unique_nodes),unique_nodes)
    # print('unique_messages',len(unique_messages),unique_messages)
    # if len(unique_messages)>0:
    #   print(unique_messages[0])
    # print('edge_info_list',edge_info_list)
    # print('unique_timestamps',len(unique_timestamps),unique_timestamps)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages) #mlp or identify

    updated_memory, updated_last_update, message_dict,memory_dict,message_trace_dict = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                                                      unique_messages,
                                                                                                                      unique_timestamps,message_dict,memory_dict,edge_info_list,message_trace_dict) #GRu ot rnn



    return updated_memory, updated_last_update,message_dict,memory_dict,message_trace_dict
  def get_updated_memory_without_contribution(self, nodes, messages):
    # Aggregate messages for the same nodes

    unique_nodes, unique_messages, unique_timestamps,edge_info_list = \
      self.message_aggregator.aggregate(
        nodes,
        messages) # last or mean


    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages) #mlp or identify

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory_without_contribution(unique_nodes,
                                                                                                                      unique_messages,
                                                                                                                      unique_timestamps) #GRu ot rnn


    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs,store_edge_info=False):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)



    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    # print('source_memory.shape()',source_memory.shape)

    # order=torch.argsort(edge_times)
    # for ii in order.tolist():
    #     i=int(ii)
    #     edge_info = {
    #         'source_node': int(source_nodes[i]),
    #         'destination_node': int(destination_nodes[i]),
    #         'edge_idx': int(edge_idxs[i]),
    #         'source_memory.shape':source_memory.shape,
    #         'destination_memory.shape':destination_memory.shape,
    #         'edge_features.shape':edge_features.shape,
    #         'time_embedding.shape':source_time_delta_encoding.shape
    #     }
    #     messages[source_nodes[i]].append((source_message[i], edge_times[i],edge_info))


    for i in range(len(source_nodes)):
        edge_info = {
            'source_node': int(source_nodes[i]),
            'destination_node': int(destination_nodes[i]),
            'edge_idx': int(edge_idxs[i]),
            'source_memory.shape':source_memory.shape,
            'destination_memory.shape':destination_memory.shape,
            'edge_features.shape':edge_features.shape,
            'time_embedding.shape':source_time_delta_encoding.shape
        }
        messages[source_nodes[i]].append((source_message[i], edge_times[i],edge_info))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder

  def set_neighbor_sampler(self, neighbor_finder):
      self.embedding_module.neighbor_sampler = neighbor_finder

  def contrast(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
               subgraph_src, subgraph_tgt, subgraph_bgd,
               explain_weights=None, edge_attr=None):

      if hasattr(self.embedding_module, 'atten_weights_list'):  # ! avoid cuda memory leakage
          self.embedding_module.atten_weights_list = []

      n_samples = len(src_idx)
      source_node_embedding, destination_node_embedding, negative_node_embedding = \
          self.get_node_emb(src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                            subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights, edge_attr)

      source_node_embedding = source_node_embedding.float()
      destination_node_embedding = destination_node_embedding.float()
      negative_node_embedding = negative_node_embedding.float()

      print('source_node_embedding',source_node_embedding.shape)
      print('destination_node_embedding',destination_node_embedding.shape)
      print('negative_node_embedding',negative_node_embedding.shape)

      score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                  torch.cat([destination_node_embedding,
                                             negative_node_embedding])).squeeze(dim=0)
      pos_score = score[:n_samples]
      neg_score = score[n_samples:]

      return pos_score, neg_score

  def contrast_node(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
               subgraph_src, subgraph_tgt, subgraph_bgd,
               explain_weights=None, edge_attr=None):

      if hasattr(self.embedding_module, 'atten_weights_list'):  # ! avoid cuda memory leakage
          self.embedding_module.atten_weights_list = []

      n_samples = len(src_idx)
      source_node_embedding, destination_node_embedding, negative_node_embedding = \
          self.get_node_emb(src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                            subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights, edge_attr)

      source_node_embedding = source_node_embedding.float()
      destination_node_embedding = destination_node_embedding.float()
      negative_node_embedding = negative_node_embedding.float()

      return source_node_embedding

  def get_node_emb(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                   subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=None, edge_attr=None):
      """
      Compute temporal embeddings for sources, destinations, and negatively sampled destinations.
      """
      n_samples = len(src_idx)
      nodes_0 = np.expand_dims(np.concatenate([src_idx, tgt_idx, bgd_idx]), axis=-1)  # [3 * bsz, 1]
      nodes_1 = np.concatenate([subgraph_src[0][0], subgraph_tgt[0][0], subgraph_bgd[0][0]], axis=0)  # [3 * bsz, n]
      nodes_2 = np.concatenate([subgraph_src[0][1], subgraph_tgt[0][1], subgraph_bgd[0][1]], axis=0)  # [3* bsz, n**2]
      node_list = [nodes_0, nodes_1, nodes_2]

      edge_1 = np.concatenate([subgraph_src[1][0], subgraph_tgt[1][0], subgraph_bgd[1][0]], axis=0)  # [3 * bsz, n]
      edge_2 = np.concatenate([subgraph_src[1][1], subgraph_tgt[1][1], subgraph_bgd[1][1]], axis=0)  # [3* bsz, n**2]
      edge_list = [edge_1, edge_2]

      time_1 = np.concatenate([subgraph_src[2][0], subgraph_tgt[2][0], subgraph_bgd[2][0]], axis=0)  # [3 * bsz, n]
      time_2 = np.concatenate([subgraph_src[2][1], subgraph_tgt[2][1], subgraph_bgd[2][1]], axis=0)  # [3* bsz, n**2]
      time_list = [time_1, time_2]
      positives = np.concatenate([src_idx, tgt_idx])

      memory = None
      time_diffs = None

      if self.use_memory:
          if self.memory_update_at_start:
              # Update memory for all nodes with messages stored in previous batches
              memory, last_update = self.get_updated_memory_without_contribution(list(range(self.n_nodes)), self.memory.messages)
          else:
              memory = self.memory.get_memory(list(range(self.n_nodes)))
              last_update = self.memory.last_update

          source_time_diffs = torch.LongTensor(cut_time).to(self.device) - last_update[
              src_idx].long()
          source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
          destination_time_diffs = torch.LongTensor(cut_time).to(self.device) - last_update[tgt_idx].long()
          destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
          negative_time_diffs = torch.LongTensor(cut_time).to(self.device) - last_update[bgd_idx].long()
          negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

          time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                                 dim=0)

      # Compute the embeddings using the embedding module
      if edge_attr is not None:
          node_embedding = self.embedding_module.embedding_update_attr(memory=memory,
                                                                       node_list=node_list,
                                                                       edge_list=edge_list,
                                                                       time_list=time_list,
                                                                       cut_time=cut_time,
                                                                       n_layers=self.num_layers,
                                                                       edge_features=edge_attr,
                                                                       explain_weights=explain_weights
                                                                       )  # [3*bs, node_feature]
      else:
          node_embedding = self.embedding_module.embedding_update(memory=memory,
                                                                  node_list=node_list,
                                                                  edge_list=edge_list,
                                                                  time_list=time_list,
                                                                  cut_time=cut_time,
                                                                  n_layers=1,
                                                                  num_neighbor=self.n_neighbors,
                                                                  explain_weights=explain_weights
                                                                  )  # [3*bs, node_feature]
      source_node_embedding = node_embedding[:n_samples]
      destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
      negative_node_embedding = node_embedding[2 * n_samples:]

      # ! We want to comment this, because want to use memory_update_at_end and don't update memory when computing scores.
      if self.use_memory and (not self.forbidden_memory_update):
          if self.memory_update_at_start:
              # Persist the updates to the memory only for sources and destinations (since now we have
              # new messages for them)
              self.update_memory(positives, self.memory.messages)

              # assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
              #     "Something wrong in how the memory was updated"

              # Remove messages for the positives since we have already updated the memory using them
              self.memory.clear_messages(positives)

          unique_sources, source_id_to_messages = self.get_raw_messages(src_idx,
                                                                        source_node_embedding,
                                                                        tgt_idx,
                                                                        destination_node_embedding,
                                                                        cut_time, e_idx)
          unique_destinations, destination_id_to_messages = self.get_raw_messages(tgt_idx,
                                                                                  destination_node_embedding,
                                                                                  src_idx,
                                                                                  source_node_embedding,
                                                                                  cut_time, e_idx)
          if self.memory_update_at_start:
              self.memory.store_raw_messages(unique_sources, source_id_to_messages)
              self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
          else:
              # ! always using memory_update_at_end
              self.update_memory(unique_sources, source_id_to_messages)
              self.update_memory(unique_destinations, destination_id_to_messages)

      return source_node_embedding, destination_node_embedding, negative_node_embedding

  def get_prob(
          self,
          src_idx_l,
          target_idx_l,
          cut_time_l,
          edge_idxs=None,
          logit=False,
          edge_idx_preserve_list=None,
          candidate_weights_dict=None,
  ):
      """
      src_idx_l, target_idx_l, cut_time_l: np.array
      edge_idxs: actually can be None... Because in self.compute_temporal_embeddings(), we will skip self.get_raw_messages() function.
      edge_idx_preserve_list: support for masking out some edges
      candidate_weights_dict: support for pg explainer

      """
      if hasattr(
              self.embedding_module, "atten_weights_list"
      ):  # ! avoid cuda memory leakage
          self.embedding_module.atten_weights_list = []

      n_samples = len(src_idx_l)
      negative_nodes = np.array(
          [
              0,
          ]
      )
      edge_idxs = None
      (
          source_node_embedding,
          destination_node_embedding,
          negative_node_embedding,
      ) = self.baseline_compute_temporal_embeddings(
          src_idx_l,
          target_idx_l,
          negative_nodes,
          cut_time_l,
          edge_idxs,
          self.n_neighbors,
          edge_idx_preserve_list=edge_idx_preserve_list,
          candidate_weights_dict=candidate_weights_dict,
      )

      score = self.affinity_score(
          torch.cat([source_node_embedding, source_node_embedding], dim=0),
          torch.cat([destination_node_embedding, negative_node_embedding]),
      ).squeeze(dim=0)
      pos_score = score[:n_samples]

      if logit:
          return pos_score
      else:
          return pos_score.sigmoid()







