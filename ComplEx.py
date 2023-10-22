class ComplEx(nn.Module):
  def __init__(self, num_entities, num_relations, embedding_dim):
    super(ComplEx, self).__init__()
    self.entity_embeddings = torch.nn.Parameter(torch.randn(num_entities, embedding_dim))
    self.relation_embeddings = torch.nn.Parameter(torch.randn(num_relations, embedding_dim))

  def forward(self):
    # return the embeddings as it is but we can regularize here by normalizing them
    return self.entity_embeddings, self.relation_embeddings 

  def ComplEx_loss(pos_edges, neg_edges, pos_reltype, neg_reltype,
                 entity_embeddings, relation_embeddings, reg=1e-3):
  # Select embeddings for both positive and negative samples
  pos_head_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 0])
  pos_tail_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 1])
  neg_head_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 0])
  neg_tail_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 1])
  pos_relation_embeds = torch.index_select(relation_embeddings, 0, pos_reltype.squeeze())
  neg_relation_embeds = torch.index_select(relation_embeddings, 0, neg_reltype.squeeze())

  # Get real and imaginary parts
  pos_re_relation, pos_im_relation = torch.chunk(pos_relation_embeds, 2, dim=1)
  neg_re_relation, neg_im_relation = torch.chunk(neg_relation_embeds, 2, dim=1)
  pos_re_head, pos_im_head = torch.chunk(pos_head_embeds, 2, dim=1)
  pos_re_tail, pos_im_tail = torch.chunk(pos_tail_embeds, 2, dim=1)
  neg_re_head, neg_im_head = torch.chunk(neg_head_embeds, 2, dim=1)
  neg_re_tail, neg_im_tail = torch.chunk(neg_tail_embeds, 2, dim=1)

  # Compute pos score
  pos_re_score = pos_re_head * pos_re_relation - pos_im_head * pos_im_relation
  pos_im_score = pos_re_head * pos_im_relation + pos_im_head * pos_re_relation
  pos_score = pos_re_score * pos_re_tail + pos_im_score * pos_im_tail
  pos_loss = -F.logsigmoid(pos_score.sum(1))


  # Compute neg score
  neg_re_score = neg_re_head * neg_re_relation - neg_im_head * neg_im_relation
  neg_im_score = neg_re_head * neg_im_relation + neg_im_head * neg_re_relation
  neg_score = neg_re_score * neg_re_tail + neg_im_score * neg_im_tail
  neg_loss = -F.logsigmoid(-neg_score.sum(1))

  loss = pos_loss + neg_loss
  reg_loss = reg * (
      pos_re_head.norm(p=2, dim=1)**2 + pos_im_head.norm(p=2, dim=1)**2 + 
      pos_re_tail.norm(p=2, dim=1)**2 + pos_im_tail.norm(p=2, dim=1)**2 +
      neg_re_head.norm(p=2, dim=1)**2 + neg_im_head.norm(p=2, dim=1)**2 + 
      neg_re_tail.norm(p=2, dim=1)**2 + neg_im_tail.norm(p=2, dim=1)**2 +
      pos_re_relation.norm(p=2, dim=1)**2 + pos_im_relation.norm(p=2, dim=1)**2 +
      neg_re_relation.norm(p=2, dim=1)**2 + neg_im_relation.norm(p=2, dim=1)**2)
  loss += reg_loss
  return loss.mean()
