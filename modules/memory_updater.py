from torch import nn
import torch
from .attribution import lrp_grucell_contrib_full, lrp_rnncell_contrib_full


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    # print('memory.get_last_update',self.memory.get_last_update(unique_node_ids))
    # print('timestamps',timestamps)

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory_without_contribution(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      # 空更新：返回原内存与时间戳，同时返回空贡献张量

      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    # print('timestamps',timestamps)
    # print(self.memory.get_last_update(unique_node_ids))

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()



    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    # print('memory contribution verify',torch.allclose(C_unique_messages.sum(dim=1) + C_updated_memory.sum(dim=1),updated_memory[unique_node_ids] , atol=1e-4))

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps





    return updated_memory, updated_last_update
  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps, C_message_full, C_memory_full,edge_info_list,C_message_trace_full):
    if len(unique_node_ids) <= 0:
      # 空更新：返回原内存与时间戳，同时返回空贡献张量

      return self.memory.memory.data.clone(), self.memory.last_update.data.clone(), {},{},{}

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()


    # num_nodes, hidden_dim = self.memory.memory.shape
    #
    # message_dim = unique_messages.shape[1] if len(unique_messages.shape) > 1 else unique_messages.shape[0]
    # memory_dim = hidden_dim



    # for node_id in range(updated_memory.shape[0]):
    #   C_message_full[node_id] = torch.ones(message_dim, memory_dim, device=unique_messages.device)
    #   C_memory_full[node_id] = torch.ones(message_dim, memory_dim, device=unique_messages.device)
    #
    #
    # print('self.memory',self.memory.memory.data)
    # print('update unique_messages',unique_messages.shape)
    # print('updated_memory',updated_memory.shape)
    # 解释一次更新：将输出贡献分配到输入

    if isinstance(self.memory_updater, nn.GRUCell):
      C_unique_messages,C_updated_memory = lrp_grucell_contrib_full(unique_messages, updated_memory[unique_node_ids],
                                                          self.memory_updater, return_ratio=True)
      print('GRU')
    elif isinstance(self.memory_updater, nn.RNNCell):
      C_unique_messages,C_updated_memory  = lrp_rnncell_contrib_full(unique_messages, updated_memory[unique_node_ids],
                                                          self.memory_updater, return_ratio=True)

      print('rnn')

    # print('C_unique_messages',C_unique_messages.shape)
    # print('C_updated_memory',C_updated_memory.shape)

    # print('memory contribution verify',C_unique_messages.sum(dim=1) + C_updated_memory.sum(dim=1))


    # print('C_unique_messages,C_updated_memory ',C_unique_messages,C_updated_messages)
    #

    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    # print('memory contribution verify',torch.allclose(C_unique_messages.sum(dim=1) + C_updated_memory.sum(dim=1),updated_memory[unique_node_ids] , atol=1e-4))

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps


    # 现在C_unique_messages和C_updated_memory已经是ratio值了
    # 不需要再除以输出值，直接使用
    for i, node_id in enumerate(unique_node_ids):
      node_id_item = node_id.item() if hasattr(node_id, 'item') else node_id
      timestamp_item = timestamps[i].item() if hasattr(timestamps[i], 'item') else timestamps[i]

      if node_id_item not in C_message_full:
        C_message_full[node_id_item] = {}
      if node_id_item not in C_memory_full:
        C_memory_full[node_id_item] = {}
      if node_id_item not in C_message_trace_full:
        C_message_trace_full[node_id_item] = {}






      edge_info = edge_info_list[i]
      message_contrib = C_unique_messages[i]  # [message_dim, memory_dim]
      memory_contrib = C_updated_memory[i]  # [memory_dim, memory_dim]



      source_dim = edge_info['source_memory.shape'][1]
      dest_dim =  edge_info['destination_memory.shape'][1]
      edge_dim =  edge_info['edge_features.shape'][1]
      time_dim =  edge_info['time_embedding.shape'][1]

      start_idx = 0
      source_contrib = message_contrib[start_idx:start_idx + source_dim, :]
      start_idx += source_dim

      dest_contrib = message_contrib[start_idx:start_idx + dest_dim, :]
      start_idx += dest_dim

      edge_contrib = message_contrib[start_idx:start_idx + edge_dim, :]
      start_idx += time_dim

      time_contrib = message_contrib[start_idx:, :]

      structured_contrib = {
        'source_node':edge_info['source_node'],
        'destination_node': edge_info['destination_node'],
        'source_node_contribution': source_contrib+memory_contrib,
        'destination_node_contribution': dest_contrib,
        'edge': edge_contrib+time_contrib,
        'edge_idx':edge_info['edge_idx']
      }


      # C_message_full[node_id_item][timestamp_item] = C_unique_messages[i]  # [1, hidden_dim]


      #
      # if timestamp_item not in C_message_full[node_id_item]:
      #   C_message_full[node_id_item][timestamp_item] = []
      #
      # edge_idx_exists = any(
      #   record.get('edge_idx') == edge_info['edge_idx'] for record in C_message_full[node_id_item][timestamp_item])
      #
      # if not edge_idx_exists:
      #
      #   C_message_full[node_id_item][timestamp_item].append(structured_contrib)

      C_message_full[node_id_item][timestamp_item] = structured_contrib  # [1, hidden_dim]

      C_memory_full[node_id_item][timestamp_item] = C_updated_memory[i]  # [1, hidden_dim]

      C_message_trace_full[node_id_item][timestamp_item]=edge_info_list[i]



    return updated_memory, updated_last_update, C_message_full,C_memory_full,C_message_trace_full


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    print('yes gru')
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    print('yes rnn')
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
