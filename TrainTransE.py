#@title Choose your model and training parameters
kg_model = "RotatE" #@param ["TransE", "ComplEx", "TransR"]
epochs = 1000 #@param {type:"slider", min:10, max:1000, step:10}
batch_size = 128 #@param {type:"number"}
learning_rate = 1e-3 #@param {type:"number"}

num_entities = 14541
num_relations = 237

if kg_model == "TransE":
    model = TransE(num_entities, num_relations, 100)
    model_loss = TransE_loss
elif kg_model == "ComplEx":
    model = ComplEx(num_entities, num_relations, 100)
    model_loss = ComplEx_loss
elif kg_model == "RotatE":
    model = RotatE(num_entities, num_relations, 50)
    model_loss = RotatE_loss
else:
    raise ValueError('Unsupported model %s' % kg_model)
