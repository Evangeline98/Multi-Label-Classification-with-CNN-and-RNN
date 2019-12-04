import pandas as pd
from torchnlp.word_to_vector import GloVe  # doctest: +SKIP
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
N_CLASSES = 1103
import pickle
def EmbeddingPar():
    vectors = GloVe()
    data = pd.read_csv("~/labelpd.csv")
    embedding = torch.zeros(N_CLASSES,300)
    print(data.shape[0])
    for i in range(data.shape[0]):
        embedding[i] = torch.sum(vectors[[data.iloc[i,j] for j in range(5) ]],dim = 0)
        n = np.sum(1-pd.isnull(data.iloc[i]))
        embedding[i]/=n
    Sim = torch.zeros(N_CLASSES,N_CLASSES)
    for i in range( N_CLASSES):
        for j in range( N_CLASSES):
            if (i>=398 and j>=398) or (i<398 and j<398):
                if i!=j:
                    Sim[i,j] = cosine_similarity(embedding[i,:].view(1,-1),embedding[j,:].view(1,-1))

    return embedding,Sim

embed,Sim = EmbeddingPar()
torch.save(embed,'/nfsshare/home/white-hearted-orange/data/em.pt')
torch.save(Sim,'/nfsshare/home/white-hearted-orange/data/Sim.pt')

#output = open('embedding.pkl','wb')
#pickle.dump(embed, output)
print(Sim)
