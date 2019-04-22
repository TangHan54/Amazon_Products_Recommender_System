from utils import model, content_model

# train content based model 
content_model.train()

# train ALS model
model.train() 
model.recommend()
model.userCf()

