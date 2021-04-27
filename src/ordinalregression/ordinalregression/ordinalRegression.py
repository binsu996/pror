import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import groupby, product
import os
from copy import deepcopy
import pickle
import inspect
import imp
import sys
from progressbar import progressbar
import shutil
import matplotlib.pyplot as plt
from .crossValidation import CrossValidation
from .evaluator import EvaluatorEX
from itertools import count
from sklearn.svm import SVR
import json

class CatDataset():
    def __init__(self,anchor_data,target_data,lower,upper):
        self.size=len(anchor_data)*len(target_data)
        self.anchors=anchor_data
        self.targets=target_data
        self.lower=lower
        self.upper=upper
        self.mod=len(anchor_data)
        self.max_diff=upper-lower
    
    def __len__(self,):
        return self.size
    
    def __getitem__(self,idx):
        i=idx//self.mod
        j=idx%self.mod
        fi,li=self.targets[i]
        fj,lj=self.anchors[j]
        label=int(li>lj)
        w=abs(li-lj)/self.max_diff
        feat=np.concatenate([fi,fj]).astype(np.float32)
        return feat,label,w


class OrdinalRegression(object):
    def __init__(self, model_protocol, model_args, config, log_dir='log',
                 model_dir=None, ignore_check_dirs=False, remove_old=False):
        # unzip the config
        self.__dict__.update(config)

        self.model = model_protocol(**model_args)
        self.model_args = model_args
        self.cfg = config
        if self.cuda:
            self.model = self.model.cuda()

        self.log_dir = log_dir
        if model_dir is not None:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(log_dir, 'model')

        if not ignore_check_dirs:
            if self.log_dir:
                if os.path.exists(self.log_dir):
                    if remove_old:
                        shutil.rmtree(self.log_dir)
                    else:
                        raise FileExistsError
                else:
                    os.makedirs(self.log_dir)
            if self.model_dir:
                if os.path.exists(self.model_dir):
                    if remove_old:
                        shutil.rmtree(self.model_dir)
                    else:
                        raise FileExistsError
                else:
                    os.makedirs(self.model_dir)

    @staticmethod
    def get_example_config():
        config = {
            'lr': 1e-3,
            'cuda': False,
            'generate_weight': True,
            'bin_width': 1,
            'batch_size': 128,
            'soft_scale': 0.5,
            'epoch': 30,
        }

    def to_pairwise_dataloader(self, anchor_data, target_data):
        anchor_feature, anchor_score = list(zip(*anchor_data))
        target_feature, target_score = list(zip(*target_data))
        upper=max(anchor_score)
        lower=min(anchor_score)
        dataset=CatDataset(anchor_data,target_data,lower,upper)
        return DataLoader(
            dataset,
            num_workers=8,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True)

    @staticmethod
    def to_pairwise_feature(anchor_features, test_features):
        N = len(anchor_features)
        M = len(test_features)
        pairwise_features = []
        pairwise_features = np.concatenate([
            np.repeat(np.reshape(test_features, [M, 1, -1]), N, axis=1),
            np.repeat(np.reshape(anchor_features, [1, N, -1]), M, axis=0),
        ], axis=-1)
        return torch.from_numpy(pairwise_features).float()

    def train(self, train_set, valid_set=None,show_time=True):
        writer = SummaryWriter(self.log_dir)
        if show_time:
            pb=progressbar
        else:
            pb=lambda x:x

        # prepare data
        train_loader = self.to_pairwise_dataloader(train_set, train_set)
        if valid_set is not None:
            valid_loader = self.to_pairwise_dataloader(train_set, valid_set)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        for step in range(self.epoch):
            train_losses, valid_losses = [], []
            train_accs, valid_accs = [], []
            best_valid_loss = None

            self.model.train()
            for x, y, w in pb(train_loader):
                opt.zero_grad()

                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    w = w.cuda()

                opt.zero_grad()

                logits = self.model(x)
                loss = loss_fn(logits*self.soft_scale, y)
                loss = torch.mean(loss*w)
                acc = torch.argmax(logits, dim=-1).eq(y).float().mean()

                train_losses.append(loss.item())
                train_accs.append(acc.item())

                loss.backward()
                opt.step()

            # torch.save(self.model, os.path.join(self.model_dir, 'lastest'))
            writer.add_scalar('train/loss', np.mean(train_losses), step)
            writer.add_scalar('train/acc', np.mean(train_accs), step)

            if valid_set is not None:
                with torch.no_grad():
                    for x, y, w in valid_loader:
                        self.model.eval()
                        if self.cuda:
                            x = x.cuda()
                            y = y.cuda()
                            w = w.cuda()
                        logits = self.model(x)
                        loss = loss_fn(logits*self.soft_scale, y)
                        loss = torch.mean(loss*w)
                        acc = torch.argmax(logits, dim=-1).eq(y).float().mean()

                        valid_losses.append(loss.item())
                        valid_accs.append(acc.item())

                if best_valid_loss is None or np.mean(valid_losses) < best_valid_loss:
                    best_valid_loss = np.mean(valid_losses)
                    self.best_param=self.model.state_dict()

                # log validation loss and acc
                writer.add_scalar('valid/loss', np.mean(valid_losses), step)
                writer.add_scalar('valid/acc', np.mean(valid_accs), step)
                writer.close()

            self.val_loss=best_valid_loss
            self.model.load_state_dict(self.best_param)

    def _get_pp_matrix(self, anchor_features, test_features):
        if hasattr(self, 'cache'):
            self.cache = []

        features = self.to_pairwise_feature(anchor_features, test_features)
        pp_matrix = []
        self.model.eval()
        with torch.no_grad():
            for feat in features:
                if self.cuda:
                    feat = feat.cuda()
                logits = self.model(feat)

                # save attention into cache
                if hasattr(self.model, 'get_attention'):
                    if not hasattr(self, 'cache'):
                        self.cache = []
                    self.cache.append(self.model.get_attention())
                pps = torch.softmax(logits, axis=-1)
                pp_matrix.append(pps[:, 1].cpu().numpy())
        return pp_matrix

    @staticmethod
    def _scoring_from_pp_matrix(pp_matrix, scores):
        preds = []
        N = len(scores)
        for pps in pp_matrix:
            idx = int(sum(pps))
            if idx >= N:
                idx = N-1
            preds.append(scores[idx])
        return preds

    def scoring(self, anchor_set, test_features, load_best_model=False):
        if load_best_model:
            self.model = torch.load(os.path.join(self.model_dir, 'best'))
        anchor_set = list(anchor_set)
        anchor_set.sort(key=lambda x: x[1])
        anchor_features, scores = list(zip(*anchor_set))
        pp_matrix = self._get_pp_matrix(anchor_features, test_features)
        preds = self._scoring_from_pp_matrix(pp_matrix, scores)
        return preds

    @staticmethod
    def _plot_pps(pps, scores):
        x = []
        y = []
        res = sorted(list(zip(pps, scores)), key=lambda x: x[1])
        for key, vals in groupby(res, lambda x: int(x[1])):
            x.append(key)
            sub_pps = list(zip(*(list(vals))))[0]
            y.append(np.mean(sub_pps)-0.5)
        fig = plt.figure()
        plt.bar(x, y)
        return fig

    @staticmethod
    def _plot_performance(preds, gts):
        x = []
        y = []
        res = sorted(list(zip(preds, gts)), key=lambda x: x[1])
        for key, vals in groupby(res, lambda x: int(x[1])):
            x.append(key)
            sub_preds = list(zip(*(list(vals))))[0]
            y.append(np.mean(sub_preds)-key)
        fig = plt.figure()
        plt.bar(x, y)
        return fig

    @staticmethod
    def _plot_attention(attentions):
        att = attentions.cpu().detach().numpy()
        fig = plt.figure()
        x = list(range(len(att)))
        img=plt.imshow(att)
        plt.colorbar(img)
        return fig

    def eval(self, anchor_set, test_set, load_best_model=False, visiable_samples=0):
        if load_best_model:
            self.model = torch.load(os.path.join(self.model_dir, 'best'))
        anchor_set = list(anchor_set)
        anchor_set.sort(key=lambda x: x[1])
        anchor_features, scores = list(zip(*anchor_set))
        test_features, gts = list(zip(*test_set))

        writer = SummaryWriter(self.log_dir)
        pp_matrix = self._get_pp_matrix(anchor_features, test_features)

        preds = self._scoring_from_pp_matrix(pp_matrix, scores)
        error_fig = self._plot_performance(preds, gts)
        writer.add_figure('error', error_fig)

        # plot some samples
        if visiable_samples > 0:
            idxs = np.random.choice(
                range(len(test_set)), size=visiable_samples)
            if hasattr(self, 'cache'):
                attention = np.array(self.cache)[idxs]
            plot_objs = np.array(pp_matrix)[idxs]
            plot_gts = np.array(gts)[idxs]
            plot_preds = np.array(preds)[idxs]
            for i in range(len(plot_objs)):
                fig = self._plot_pps(plot_objs[i], scores)
                writer.add_figure('sample/gt{}_pred{}'.format(
                    plot_gts[i], plot_preds[i]), fig)
                if hasattr(self, 'cache'):
                    fig = self._plot_attention(attention[i])
                    writer.add_figure('attention/gt{}_pred{}'.format(
                        plot_gts[i], plot_preds[i]), fig)

        metrics = EvaluatorEX.eval(preds, gts, 1)
        writer.add_hparams(self.cfg, metrics)
        writer.close()
        return metrics,preds

    def release_with_anchors(self, anchors, release_file, load_best_model=False):
        anchors.sort(key=lambda x: x[1])
        anchors = list(zip(*anchors))
        if load_best_model:
            model = torch.load(os.path.join(self.model_dir, 'best'))
        else:
            model = self.model
        model = model.cpu()
        type_name = type(model).__name__
        src, params = self.serialization(model)
        model_args = self.model_args
        pickle.dump([src, type_name, model_args, params, anchors],
                    open(release_file, 'wb'))
        return model
    
    def _save_export_files(self,anchors,test,release_file,load_best_model):
        anchors.sort(key=lambda x: x[1])
        if load_best_model:
            model = torch.load(os.path.join(self.model_dir, 'best'))
        else:
            model = self.model
        model = model.cpu()
        model.eval()

        if not os.path.exists(release_file):
            os.makedirs(release_file)

        # write test result
        metrics,preds=self.eval(anchors,test,load_best_model=True)
             
        with open(os.path.join(release_file,'test'),'w+') as f:
            for (x,y),p in zip(test,preds):
                f.write('{} {} {}\n'.format(str(p), str(y),' '.join(list(map(str,x)))))

        json.dump(metrics,open(os.path.join(release_file,'report.json'),'w+'))

        # write anchor
        with open(os.path.join(release_file,'anchors'),'w+') as f:
            for x,y in anchors:
                f.write('{} {}\n'.format(str(y),' '.join(list(map(str,x)))))
        return model
    
    def release_cpp_model_with_anchor(self,anchors,test,release_file,load_best_model=False):
        # save model
        model=self._save_export_files(anchors,test,release_file,load_best_model)
        D=np.array(anchors[0][0]).shape[-1]
        example=torch.Tensor(np.ndarray([1,D*2]))
        cpp_model=torch.jit.trace(model,example)
        cpp_model.save(os.path.join(release_file,'model'))

    def release_onnx_model_with_anchor(self,anchors,test,release_file,load_best_model=False):
        model=self._save_export_files(anchors,test,release_file,load_best_model)
        D=np.array(anchors[0][0]).shape[-1]
        example=torch.Tensor(np.ndarray([1,D*2]))
        torch.onnx.export(model,               # model being run
                example,                         # model input (or a tuple for multiple inputs)
                os.path.join(release_file,'model'),   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},
                opset_version=10,
        )


    @staticmethod
    def serialization(model):
        source_file = inspect.getfile(type(model))
        str_source = open(source_file, 'r').read()
        assert isinstance(model, torch.nn.Module)
        params = model.state_dict()
        return str_source, params

    @staticmethod
    def load_release_model(file):
        src, type_name, args, params, anchors = pickle.load(open(file, 'rb'))
        mod = imp.new_module('TMP')
        exec(src, mod.__dict__,)
        definition_obj = getattr(mod, type_name)
        model = definition_obj(**args)
        assert isinstance(model, torch.nn.Module)
        model.load_state_dict(params)
        return model, anchors

    def set_model(self, model):
        self.model = model
        if self.cuda:
            self.model = self.model.cuda()


class CVTrial(object):
    def __init__(self, cv_args, ordinal_regression_args):
        self.cv_args = cv_args
        self.ordinal_regression_args = ordinal_regression_args
        self.home_log_dir = self.ordinal_regression_args['log_dir']
        del self.ordinal_regression_args['log_dir']

    @staticmethod
    def get_example_cv_args():
        return {
            'data': 'place your data here, [[feature,label] ... ] is expected',
            'n_folds': 5,
            'shuffle_seed': 888,
            'split_seed': 666
        }

    @staticmethod
    def get_example_ordinal_regression_args():
        return {
            'model_protocol': 'the model type which will be init',
            'model_args': 'place the param to init the model',
            'config': OrdinalRegression.get_example_config(),
            'log_dir': 'log',
            'ignore_check_dirs': False,
            'remove_old': False,
        }

    def run(self):
        cv = CrossValidation(**self.cv_args)
        writer = SummaryWriter(self.home_log_dir)
        labels = list(zip(*self.cv_args['data']))[1]
        distribution = plt.figure()
        plt.hist(labels, bins=100, edgecolor="black")
        writer.add_figure('distribution', distribution)
        metric_logger = EvaluatorEX()
        fold = count(1)
        for train_set, valid_set, test_set in cv:
            log_dir = "{}/{}th".format(self.home_log_dir, next(fold))
            org = OrdinalRegression(
                log_dir=log_dir, **self.ordinal_regression_args)
            org.train(train_set, valid_set)
            metircs,_ = org.eval(train_set, test_set, load_best_model=True,
                               visiable_samples=50)
            metric_logger.add_record(metircs)
        average_metircs = metric_logger.get_current_mean()
        writer.add_hparams(self.ordinal_regression_args['config'],
                           average_metircs)

class Releaser(object):

    def __init__(self, ordinal_regression_args):
        self.ordinal_regression_args = ordinal_regression_args
        self.home_log_dir = self.ordinal_regression_args['log_dir']
        del self.ordinal_regression_args['log_dir']
    
    def run(self,data,test,release_file,save_format='python'):
        writer = SummaryWriter(self.home_log_dir)
        org = OrdinalRegression(
            log_dir=self.home_log_dir, **self.ordinal_regression_args)
        org.train(data, data)
        if save_format=='python':
            org.release_with_anchors(data,release_file=release_file,load_best_model=True)
        elif save_format=='cpp':
            org.release_cpp_model_with_anchor(data,test,release_file=release_file,load_best_model=True)
        elif save_format=='onnx':
            org.release_onnx_model_with_anchor(data,test,release_file=release_file,load_best_model=True)
        else:
            raise ValueError
