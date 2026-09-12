"""Fixed-epoch, train-only neural fitting with portable state dictionaries."""
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from copper_mvp.common import WorkbenchError
from copper_mvp.neural_runtime import tensor_random_scope
from copper_mvp.tabular_runtime import tabular_runtime
from copper_mvp.tabular_registry import TRAINING,ARCHITECTURES,training_parameters
from copper_mvp.tabular_inputs import NeuralInputs


def batches(order,size):
    pieces=list(order.split(size))
    if len(pieces)>1 and len(pieces[-1])==1:
        pieces[-2]=order[-(len(pieces[-2])+1):]
        pieces.pop()
    return pieces


class TabularModel:
    def __init__(self,method,seed,training=None,reference=False):
        self.method_id,self.seed=method,seed
        self.training=dict(training_parameters(method) if training is None else training)
        self.architecture=dict(ARCHITECTURES[method])
        self.fit_warnings=[]
        self.reference=reference
        self.network_method=method

    def fit(self,X,y,sample_weight=None,max_wall_seconds=None):
        torch=tabular_runtime()
        from copper_mvp.tabular_networks import TabularNetwork,matched_mlp_architecture
        X,y=np.asarray(X,float),np.asarray(y,float)
        self.architecture=dict(ARCHITECTURES[self.method_id])
        self.preprocessor=NeuralInputs()
        numerical,categories=self.preprocessor.fit_transform(X)
        self.target_scaler=StandardScaler().fit(y-X[:,:2])
        target=self.target_scaler.transform(y-X[:,:2]).astype(np.float32)
        weight=np.ones(len(X),np.float32) if sample_weight is None else np.asarray(sample_weight,np.float32)
        tensors=(torch.from_numpy(numerical),torch.from_numpy(categories),torch.from_numpy(target),torch.from_numpy(weight))
        started=time.perf_counter();history=[];updates=0;skipped=0
        with tensor_random_scope(torch,self.seed):
            network=TabularNetwork(self.method_id,self.architecture)
            parameter_budget=sum(p.numel() for p in network.parameters())
            if self.reference:
                matched=matched_mlp_architecture(parameter_budget)
                self.architecture={"hidden":matched["hidden"]}
                self.network_method="MLPControl"
                network=TabularNetwork(self.network_method,self.architecture)
            network.initialize(tensors[0],tensors[1])
            initial={name:parameter.detach().clone() for name,parameter in network.named_parameters()}
            optimizer=torch.optim.AdamW(network.parameters(),lr=self.training["learning_rate"],
                                         weight_decay=self.training["weight_decay"])
            generator=torch.Generator().manual_seed(self.seed)
            for epoch in range(self.training["epochs"]):
                network.train();loss_sum=0.;seen_weight=0.
                for indices in batches(torch.randperm(len(X),generator=generator),self.training["batch_size"]):
                    if max_wall_seconds is not None and time.perf_counter()-started>max_wall_seconds:
                        raise WorkbenchError("神经模型达到训练预算","NEURAL_TIME_BUDGET")
                    total_weight=tensors[3][indices].sum()
                    if total_weight.item()==0:
                        skipped+=1
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    prediction,penalty=network(tensors[0][indices],tensors[1][indices])
                    per_row=(prediction-tensors[2][indices]).square().mean(dim=1)
                    objective=(per_row*tensors[3][indices]).sum()/total_weight+penalty
                    if not torch.isfinite(objective):raise WorkbenchError("神经模型损失非有限","NEURAL_LOSS")
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(),self.training["gradient_clip"])
                    optimizer.step();updates+=1
                    loss_sum+=float((per_row.detach()*tensors[3][indices]).sum())
                    seen_weight+=float(total_weight)
                history.append({"epoch":epoch+1,"weighted_standardized_mse":loss_sum/seen_weight})
            network.eval()
            self.state={name:value.detach().cpu().clone() for name,value in network.state_dict().items()}
            changes={name:float((parameter.detach()-initial[name]).abs().max())
                     for name,parameter in network.named_parameters()}
            self._network=network
        self.models=[{"method":self.method_id,"state_dict":self.state}]
        self.fit_metadata={"training_rows":len(X),"numeric_features":220,"categorical_features":4,
            "missing_mask_features":110,"target_delta_mean":self.target_scaler.mean_.tolist(),
            "target_delta_scale":self.target_scaler.scale_.tolist(),"epochs":self.training["epochs"],
            "optimizer_updates":updates,"zero_weight_batches_skipped":skipped,"training_history":history,
            "parameters":sum(p.numel() for p in network.parameters()),"parameter_max_changes":changes,
            "elapsed_seconds":time.perf_counter()-started,"device":"cpu","native_threads":1,
            "validation_used":False,"early_stopping":False,"prediction_mode":"eval",
            "role":"parameter_matched_mlp_control" if self.reference else "candidate",
            "reference_for":self.method_id if self.reference else None,
            "parameter_budget":parameter_budget,
            "parameter_budget_relative_difference":abs(sum(p.numel() for p in network.parameters())-parameter_budget)/parameter_budget}
        return self

    def __getstate__(self):
        state=self.__dict__.copy()
        state.pop("_network",None)
        return state

    def network(self):
        if not hasattr(self,"_network"):
            torch=tabular_runtime()
            from copper_mvp.tabular_networks import TabularNetwork,matched_mlp_architecture
            with tensor_random_scope(torch,self.seed):
                self._network=TabularNetwork(self.network_method,self.architecture)
            self._network.load_state_dict(self.state)
            self._network.eval()
        return self._network

    def predict(self,X):
        torch=tabular_runtime()
        X=np.asarray(X,float)
        numerical,categories=self.preprocessor.transform(X)
        network=self.network();network.eval()
        predictions=[]
        with torch.inference_mode():
            for start in range(0,len(X),128):
                values,_=network(torch.from_numpy(numerical[start:start+128]),torch.from_numpy(categories[start:start+128]))
                predictions.append(values.numpy())
        delta=np.concatenate(predictions,axis=0)
        return X[:,:2]+self.target_scaler.inverse_transform(delta)
