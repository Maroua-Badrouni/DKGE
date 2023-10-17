class TransR(nn.Module):
  def __init__(self, num_entities, num_relations, embedding_dim):
    super(TransR, self).__init__()
    # entity embeddings has equal real and imaginary parts, so we double the dimension size
    self.entity_embeddings = torch.nn.Parameter(torch.randn(num_entities, 2*embedding_dim))
    self.relation_embeddings = torch.nn.Parameter(torch.randn(num_relations, embedding_dim))

  def forward(self):
    # return the embeddings as it is but we can regularize here by normalizing them
    return self.entity_embeddings, self.relation_embeddings

def TransR_loss(pos_edges, neg_edges, pos_reltype, neg_reltype, entity_embeddings, relation_embeddings, 
                gamma=5.0, epsilon=2.0):
  # Select embeddings for both positive and negative samples
  pos_head_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 0])
  pos_tail_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 1])
  neg_head_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 0])
  neg_tail_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 1])
  pos_relation_embeds = torch.index_select(relation_embeddings, 0, pos_reltype.squeeze())
  neg_relation_embeds = torch.index_select(relation_embeddings, 0, neg_reltype.squeeze())

  # Dissect the embedding in equal chunks to get real and imaginary parts
  pos_re_head, pos_im_head = torch.chunk(pos_head_embeds, 2, dim=1)
  pos_re_tail, pos_im_tail = torch.chunk(pos_tail_embeds, 2, dim=1)
  neg_re_head, neg_im_head = torch.chunk(neg_head_embeds, 2, dim=1)
  neg_re_tail, neg_im_tail = torch.chunk(neg_tail_embeds, 2, dim=1)

  # Make phases of relations uniformly distributed in [-pi, pi]
  embedding_range = 2 * (gamma + epsilon) / pos_head_embeds.size(-1)
  pos_phase_relation = pos_relation_embeds/(embedding_range/np.pi)

  pos_re_relation = torch.cos(pos_phase_relation)
  pos_im_relation = torch.sin(pos_phase_relation)

  neg_phase_relation = neg_relation_embeds/(embedding_range/np.pi)
  neg_re_relation = torch.cos(neg_phase_relation)
  neg_im_relation = torch.sin(neg_phase_relation)


  # Compute pos score
  pos_re_score = pos_re_head * pos_re_relation - pos_im_head * pos_im_relation
  pos_im_score = pos_re_head * pos_im_relation + pos_im_head * pos_re_relation
  pos_re_score = pos_re_score - pos_re_tail 
  pos_im_score = pos_im_score - pos_im_tail
  # Stack and take squared norm of real and imaginary parts
  pos_score = torch.stack([pos_re_score, pos_im_score], dim = 0)
  pos_score = pos_score.norm(dim = 0)
  # Log sigmoid of margin loss
  pos_score = gamma - pos_score.sum(dim = 1)
  pos_score = - F.logsigmoid(pos_score)

  # Compute neg score
  neg_re_score = neg_re_head * neg_re_relation - neg_im_head *neg_im_relation
  neg_im_score = neg_re_head * neg_im_relation + neg_im_head * neg_re_relation
  neg_re_score = neg_re_score - neg_re_tail 
  neg_im_score = neg_im_score - neg_im_tail
  # Stack and take squared norm of real and imaginary parts
  neg_score = torch.stack([neg_re_score, neg_im_score], dim = 0)
  neg_score = neg_score.norm(dim = 0)
  # Log sigmoid of margin loss
  neg_score = gamma - neg_score.sum(dim = 1)
  neg_score = - F.logsigmoid(-neg_score)

  loss = (pos_score + neg_score)/2
  
  return loss.mean()
