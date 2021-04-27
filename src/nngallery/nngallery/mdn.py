import torch
import numpy as np

class MixtureDensityNetwork(torch.nn.Module):
    def __init__(self,n_components,hiddenlayers,hidden_size):
        super().__init__()
        self.hiddenlayers=hiddenlayers
        self.n_components=n_components
        self.output=torch.nn.Linear(hidden_size,3*n_components)
    
    def forward(self,x,y=None,how='max'):
        x=self.hiddenlayers(x)
        x=self.output(x)
        weight,mu,sigma=torch.split(x,self.n_components,dim=1)
        if y is not None:
            sigma=torch.exp(sigma).clip(min=1e-5)
            coef=1/sigma/((2*np.pi)**0.5)
            expv=torch.exp(-(y.reshape(-1,1)-mu)**2/(2*sigma**2))
            weight=torch.softmax(weight,dim=1)
            prob=coef*expv*weight
            loss=torch.mean(-torch.log(torch.sum(prob,dim=1)))
            return loss
        else:
            if how=='max':
                index0=torch.arange(0,x.size()[0])
                index1=torch.argmax(weight,dim=1)
                print(index1)
                return mu[index0.long(),index1.long()]
            if how=='mean':
                return (mu*weight).sum(dim=1)

    def train(self,train_X,train_y,epoch=16,cuda=True,batch_size=128,lr=1e-4):
        opt=torch.optim.Adam(self.parameters(),lr=lr)
        train_dataset=torch.utils.data.TensorDataset(torch.Tensor(train_X),torch.Tensor(train_y))
        train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

        if cuda:
            self.cuda()
        
        for i in range(1,epoch+1):
            for x,y in train_dataloader:
                if cuda:
                    x,y=x.cuda(),y.cuda()
                opt.zero_grad()
                loss=self.forward(x,y)
                loss.backward()
                opt.step()
                print(f'{i}-th loss: {loss.item():.3f}')
        
        self.cpu()
        

    def predict(self,test_X,batch_size=None,cuda=True,how='max'):

        if batch_size is not None:
            raise NotImplementedError
        if cuda:
            self.cuda()
        with torch.no_grad():
            x=torch.Tensor(test_X)
            if cuda:
                x=x.cuda()
            prediction=self.forward(x,how=how)

        self.cpu()

        return prediction.cpu().numpy()

        
        

        
                  


        

            


