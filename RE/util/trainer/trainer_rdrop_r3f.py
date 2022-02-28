import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import logging
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, accuracy_score
logging.basicConfig(filename='./eval.log', level=logging.INFO)

#from ranger import Ranger  # this is from ranger.py
# from ranger import RangerVA  # this is from ranger913A.py
# from ranger import RangerQH  # this is from rangerqh.py

class Trainer_rdrop_r3f:
    def __init__(self, task, model,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=1000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, distributed = False,
                 local_rank = 0, accum_iter=1, seed = None, model_name=None):
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:2" if cuda_condition else "cpu")
        self.seed = seed
        self.model = model
        self.model_name = model_name
        self.task = task
        self.local_rank = local_rank
        self.distributed = distributed

        self.accum_iter = accum_iter

        # if self.local_rank == 0:
        #     self.writer = SummaryWriter()
        self.avgloss = 0

        self.now_iteration = 0
        self.max_acc = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # DDP
        if distributed:
            self.model.cuda()
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            self.device = torch.device(
                f"cuda:{local_rank}"  # if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
            )
        # nn.DataParallel if CUDA can detect more than 1 GPU
        elif with_cuda and torch.cuda.device_count() > 1:
            self.model = self.model.to(self.device)
            print("Using %d GPUS for your model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=[0,1,2,3])
        else: # if gpu = 1 or not gpu
            self.model = self.model.to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]

        # Each of the Ranger, RangerVA, RangerQH have different parameters.
        # self.optim = Ranger(optimizer_grouped_parameters, lr=lr)

        self.optim = AdamW(optimizer_grouped_parameters, lr=lr, betas=betas)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim,
            num_warmup_steps=1000,
            num_training_steps=8000)
        
        self.noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=1e-5)
        self.r3f_lambda = 1.0

        self.log_freq = log_freq

    def train(self, epoch):
        self.iteration(epoch, self.train_data)
        
    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
        ) / noised_logits.size(0)

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')       

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
    
    def iteration(self, epoch, data_loader, train=True):
        
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0


        for i, data in data_iter:
            self.model.train()
            if train:
                self.now_iteration += 1

            data = {key: value.to(self.device) for key, value in data.items()}
            
            output = self.model.forward(**data)
            loss_1, logits_1 = output.loss, output.logits
                        
            #####           
            output = self.model.forward(**data)
            loss_2, logits_2 = output.loss, output.logits
       
            # cross entropy loss for classifier
            ce_loss = 0.5 * (loss_1 + loss_2)

            kl_loss = self.compute_kl_loss(logits_1, logits_2)

            # carefully choose hyper-parameters
            a = 0.7
            rdrop_loss = ce_loss + a * kl_loss            
            #####
            
            token_embeddings = self.model.module.model.embeddings(input_ids=data['input_ids'])
            
            output = self.model.forward(input_ids=None, attention_mask=data["attention_mask"], 
                                        span_1=data["span_1"], span_2=data["span_2"], labels=data["labels"], inputs_embeds=token_embeddings)
            
            noise = self.noise_sampler.sample(sample_shape=token_embeddings.shape).to(token_embeddings)
            noised_embeddings = token_embeddings.detach().clone() + noise

            noised_output = self.model(
                input_ids=None,
                inputs_embeds=noised_embeddings,
                attention_mask=data['attention_mask'],
                span_1=data['span_1'],
                span_2=data['span_2'],
                labels=data['labels']
            )
            symm_kl = self._get_symm_kl(noised_output.logits, output.logits)
            ####################

            r3f_loss =  output.loss + (self.r3f_lambda * symm_kl)
            
            loss = rdrop_loss + r3f_loss
        

            predict = F.softmax(output.logits, dim=1).argmax(dim=1)
            correct = (predict == data['labels'].long()).sum().item()

            acc = correct / len(data["labels"]) * 100

            ##
            accum_iter = self.accum_iter
            loss = loss.mean() / accum_iter
            loss.backward()

            if accum_iter == 1:  # not gradient accumulation
                self.optim.step()
                self.scheduler.step()
                self.optim.zero_grad()
            elif ((i + 1) % accum_iter == 0) or (i + 1 == len(data_iter)):  # gradient accumulation
                self.optim.step()
                self.scheduler.step()
                self.optim.zero_grad()


            if self.distributed == False:
                post_fix = {
                    "epoch": epoch,
                    "iter": self.now_iteration,
                    "loss": loss,
                    "train_acc": acc
                }

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

            elif self.distributed == True and self.local_rank == 0 and train:
                post_fix = {
                    "epoch": epoch,
                    "iter": self.now_iteration,
                    "avg_loss": self.avgloss / (self.now_iteration),
                    "loss": loss.item(),
                    "train_acc" : acc
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

            if self.now_iteration % 200 == 0 and self.now_iteration >= 3000:
                # eval
                eval_acc = self.evaluation(epoch, self.test_data, train=False)
                self.model.train
                # save
                output_path = "output/" + str(self.now_iteration) +"_"+str(eval_acc)[:5] + '_' + str(self.seed) \
                              + "_" + self.model_name

                if eval_acc > self.max_acc:
                    self.max_acc = eval_acc
                    torch.save(self.model.module.state_dict(), output_path) # save state dict -> for wic task
                    print("EP:%d Model Saved on:" % epoch, output_path)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))


############
    def evaluation(self, epoch, data_loader, train=False):
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        n_correct = 0
        data_len = 0
        self.model.eval()
        torch.cuda.empty_cache()

        gold_list = []
        pred_list = []

        with torch.no_grad():
            for i, data in data_iter:

                data = {key: value.to(self.device) for key, value in data.items()}

                output = self.model.forward(input_ids=data['input_ids'], attention_mask=data["attention_mask"], 
                                        span_1=data["span_1"], span_2=data["span_2"], labels=data["labels"])
                predict = F.softmax(output.logits, dim=1).argmax(dim=1)

                correct = (predict == data['labels'].long()).sum().item()


                n_correct += correct
                data_len += len(data["labels"])

                gold_list += data['labels'].long().tolist()
                pred_list += predict.tolist()


        eval_acc = n_correct / data_len * 100
        print("eval acuracy = " + str(eval_acc))
        logging.info("eval acuracy = " + str(eval_acc)[:5] + " iter: "+ str(self.now_iteration))
        logging.info('Matthew\'s Corr : ' + str(matthews_corrcoef(gold_list, pred_list)))

        print(n_correct)
        print(data_len)

        torch.cuda.empty_cache()

        return eval_acc
