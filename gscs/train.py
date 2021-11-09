import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import threading
import matplotlib.pyplot as plt

import utils
import config
import data
import second
import eval
import pickle



class Train(object):

    def __init__(self, vocab_file_path=None, model_file_path=None):
        """

        :param vocab_file_path: tuple of source vocab, code vocab, ast vocab, nl vocab,
                                if given, build vocab by given path
        :param model_file_path:
        """

        # dataset
        self.train_dataset = data.CodePtrDataset(code_path=config.train_code_path,type_path=config.train_type_path,nl_path=config.train_nl_path)
        self.train_dataset_size = len(self.train_dataset)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           collate_fn=lambda *args: utils.collate_fn(args,code_vocab=self.code_vocab,type_vocab=self.type_vocab,nl_vocab=self.nl_vocab))

        # vocab
        self.code_vocab: utils.Vocab
        self.type_vocab: utils.Vocab
        self.nl_vocab: utils.Vocab
        # load vocab from given path
        if vocab_file_path:
            code_vocab_path, type_vocab_path, nl_vocab_path = vocab_file_path
            self.code_vocab = utils.load_vocab_pk(code_vocab_path)
            self.type_vocab = utils.load_vocab_pk(type_vocab_path)
            self.nl_vocab = utils.load_vocab_pk(nl_vocab_path)
        # new vocab
        else:
            self.code_vocab = utils.Vocab('code_vocab')
            self.type_vocab = utils.Vocab('type_vocab')
            self.nl_vocab = utils.Vocab('nl_vocab')
            codes, types, nls = self.train_dataset.get_dataset()
            for code, typei, nl in zip(codes, types, nls):
                self.code_vocab.add_sentence(code)
                self.type_vocab.add_sentence(typei)
                self.nl_vocab.add_sentence(nl)

            self.origin_code_vocab_size = len(self.code_vocab)
            self.origin_type_vocab_size = len(self.type_vocab)
            self.origin_nl_vocab_size = len(self.nl_vocab)

            # trim vocabulary
            self.code_vocab.trim(config.code_vocab_size)
            self.type_vocab.trim(config.type_vocab_size)
            self.nl_vocab.trim(config.nl_vocab_size)

            # save vocabulary
            self.code_vocab.save(config.code_vocab_path)
            self.type_vocab.save(config.type_vocab_path)
            self.nl_vocab.save(config.nl_vocab_path)
            self.code_vocab.save_txt(config.code_vocab_txt_path)
            self.type_vocab.save_txt(config.type_vocab_txt_path)
            self.nl_vocab.save_txt(config.nl_vocab_txt_path)

        self.code_vocab_size = len(self.code_vocab)
        self.type_vocab_size = len(self.type_vocab)
        self.nl_vocab_size = len(self.nl_vocab)

        # model
        self.model = second.Model(code_vocab_size=self.code_vocab_size, type_vocab_size=self.type_vocab_size, nl_vocab_size=self.nl_vocab_size, model_file_path=model_file_path)
        self.params = list(self.model.gat_encoder.parameters()) + list(self.model.code_encoder.parameters()) + list(self.model.decoder.parameters())

        # optimizer
        self.optimizer = Adam([
            {'params': self.model.gat_encoder.parameters(), 'lr': config.gat_encoder_lr},
            {'params': self.model.code_encoder.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.decoder.parameters(), 'lr': config.decoder_lr},
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        if config.use_lr_decay:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=config.lr_decay_every,
                                                    gamma=config.lr_decay_rate)
        
        # if model_file_path:
        #     self.optimizer.load_state_dict(torch.load(model_file_path)['optimizer'])
        #     self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,step_size=config.lr_decay_every,gamma=config.lr_decay_rate,last_epoch=13)
        
        # best score and model(state dict)
        self.max_bleu: float = 0
        #self.min_loss: float = 1000
        self.best_model: dict = {}
        self.best_epoch_batch: (int, int) = (None, None)

        # eval instance
        self.eval_instance = eval.Eval(self.get_cur_state_dict())

        # early stopping
        self.early_stopping = None
        if config.use_early_stopping:
            self.early_stopping = utils.EarlyStopping()

        # creates the model dir for this run
        config.model_dir = os.path.join(config.model_dir, utils.get_timestamp())
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)

    def run_train(self):
        """
        start training
        :return:
        """
        self.train_iter()
        return self.best_model

    def get_batch(self, edge_data, idx, bs):
        tmp = edge_data.iloc[idx*config.batch_size: idx*config.batch_size+bs]
        x1 = []
        for _, item in tmp.iterrows():
            x1.append(item['adj'])
        return x1

    def train_one_batch(self, batch: utils.Batch, edge_batch, batch_size, criterion):
        """
        train one batch
        :param batch: get from collate_fn of corresponding dataloader, class Batch
        :param batch_size: batch size
        :param criterion: loss function
        :return: avg loss
        """
        nl_batch = batch.extend_nl_batch if config.use_pointer_gen else batch.nl_batch
        #print(nl_batch.size())

        self.optimizer.zero_grad()

        decoder_outputs = self.model(batch, batch_size, self.nl_vocab, edge_batch)     # [T, B, nl_vocab_size]
        #print(decoder_outputs.size())

        batch_nl_vocab_size = decoder_outputs.size()[2]     # config.nl_vocab_size (+ max_oov_num)
        decoder_outputs = decoder_outputs.view(-1, batch_nl_vocab_size)
        nl_batch = nl_batch.view(-1)
        #print(nl_batch.size())
        #print(decoder_outputs.size())
        

        loss = criterion(decoder_outputs, nl_batch)
        loss.backward()

        # address over fit梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.params, 5)

        self.optimizer.step()

        return loss

    def train_iter(self):
        start_time = time.time()
        plot_losses = []

        criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))

        for epoch in range(0, 50):
            print_loss = 0
            plot_loss = 0
            last_print_index = 0
            last_plot_index = 0
            i = 0
            for index_batch, batch in enumerate(self.train_dataloader):


                batch_size = batch.batch_size
                if index_batch % 272 == 0 :
                    adj_path = 'train/code_original'+str(i)+".pkl"
                    edge_file = open(config.edge_dir+adj_path,'rb')
                    edge_data = pickle.load(edge_file)
                    edge_file.close()
                    i = i + 1
                    index = 0

                edge_batch = self.get_batch(edge_data, index, batch_size)
                loss = self.train_one_batch(batch, edge_batch, batch_size, criterion)
                '''
                try:
                    loss = self.train_one_batch(batch, edge_batch, batch_size, criterion)
                except:
                    print(index_batch)
                    print(index)
                    index = index + 1
                    continue
                '''
            
                print_loss += loss.item()
                plot_loss += loss.item()
                index = index + 1


                # print train progress details
                if index_batch % config.print_every == 0:
                    cur_time = time.time()
                    utils.print_train_progress(start_time=start_time, cur_time=cur_time, epoch=epoch,
                                               n_epochs=config.n_epochs, index_batch=index_batch, batch_size=batch_size,
                                               dataset_size=self.train_dataset_size, loss=print_loss,
                                               last_print_index=last_print_index)
                    print_loss = 0
                    last_print_index = index_batch

                # plot train progress details
                if index_batch % config.plot_every == 0:
                    batch_length = index_batch - last_plot_index
                    if batch_length != 0:
                        plot_loss = plot_loss / batch_length
                    plot_losses.append(plot_loss)
                    plot_loss = 0
                    last_plot_index = index_batch

                # save check point
                if config.use_check_point and index_batch % config.save_check_point_every == 0:
                    pass

                # validate on the valid dataset every config.valid_every batches
                
                if config.validate_during_train and index_batch % config.validate_every == 0 and index_batch != 0:
                    print('\nValidating the model at epoch {}, batch {} on valid dataset......'.format(
                        epoch, index_batch))
                    config.logger.info('Validating the model at epoch {}, batch {} on valid dataset.'.format(
                        epoch, index_batch))
                    self.valid_state_dict(state_dict=self.get_cur_state_dict(), epoch=epoch, batch=index_batch)

                    if config.use_early_stopping:
                        if self.early_stopping.early_stop:
                            break
                
            
            
            if config.use_early_stopping:
                if self.early_stopping.early_stop:
                    break
            

            # validate on the valid dataset every epoch
            if config.validate_during_train:
                print('\nValidating the model at the end of epoch {} on valid dataset......'.format(epoch))
                config.logger.info('Validating the model at the end of epoch {} on valid dataset.'.format(epoch))
                self.valid_state_dict(self.get_cur_state_dict(), epoch=epoch)
                if config.use_early_stopping:
                    if self.early_stopping.early_stop:
                        break

            if config.use_lr_decay:
                self.lr_scheduler.step()
        
        plt.xlabel('every {} batches'.format(config.plot_every))
        plt.ylabel('avg loss')
        plt.plot(plot_losses)
        plt.savefig(os.path.join(config.out_dir, 'train_loss_{}.svg'.format(utils.get_timestamp())),
                    dpi=600, format='svg')
        utils.save_pickle(plot_losses, os.path.join(config.out_dir, 'plot_losses_{}.pk'.format(utils.get_timestamp())))

        # save the best model
        if config.save_best_model:
            best_model_name = 'best_epoch-{}_batch-{}.pt'.format(
                self.best_epoch_batch[0], self.best_epoch_batch[1] if self.best_epoch_batch[1] != -1 else 'last')
            self.save_model(name=best_model_name, state_dict=self.best_model)

    def save_model(self, name=None, state_dict=None):
        """
        save current model
        :param name: if given, name the model file by given name, else by current time
        :param state_dict: if given, save the given state dict, else save current model
        :return:
        """
        if state_dict is None:
            state_dict = self.get_cur_state_dict()
        if name is None:
            model_save_path = os.path.join(config.model_dir, 'model_{}.pt'.format(utils.get_timestamp()))
        else:
            model_save_path = os.path.join(config.model_dir, name)
        torch.save(state_dict, model_save_path)

    def save_check_point(self):
        pass

    def get_cur_state_dict(self) -> dict:
        """
        get current state dict of model
        :return:
        """
        state_dict = {
                'code_encoder': self.model.code_encoder.state_dict(),
                'gat_encoder': self.model.gat_encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        return state_dict

    def valid_state_dict(self, state_dict, epoch, batch=-1):
        self.eval_instance.set_state_dict(state_dict)
        #loss, bleu = self.eval_instance.run_eval()
        bleu = self.eval_instance.run_eval()

        if config.save_valid_model:
            #model_name = 'model_valid-loss-{:.4f}_epoch-{}_batch-{}.pt'.format(loss, epoch, batch)
            model_name = 'model_valid-bleu-{:.4f}_epoch-{}_batch-{}.pt'.format(bleu, epoch, batch)
            save_thread = threading.Thread(target=self.save_model, args=(model_name, state_dict))
            save_thread.start()
        
        if bleu > self.max_bleu:
            self.max_bleu = bleu
            print("the current max_bleu {:.4f}".format(self.max_bleu))
            self.best_model = state_dict
            self.best_epoch_batch = (epoch, batch)
        if config.use_early_stopping:
            self.early_stopping(bleu)

        '''
        if loss < self.min_loss:
            self.min_loss = loss
            self.best_model = state_dict
            self.best_epoch_batch = (epoch, batch)

        if config.use_early_stopping:
            self.early_stopping(loss)

        '''
        #if loss < self.min_loss:
           # self.min_loss = loss

        

