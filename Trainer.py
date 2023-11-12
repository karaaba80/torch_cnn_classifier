import torch
from torch import nn, optim
import copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time_utils
import os
import shutil

import evaluator

def svm_loss(output, target):
    margin = 1  # Margin for the SVM loss
    num_classes = output.size(1)

    # Construct matrix of correct scores
    correct_scores = output[torch.arange(output.size(0)), target].view(-1, 1)

    # Calculate SVM loss
    loss = torch.sum(torch.clamp(output - correct_scores + margin, min=0))
    loss -= margin  # Subtract the margin for the correct class

    # Average the loss
    loss /= output.size(0)

    return loss

class trainer:
    def __init__(self, model, lrate=0.0075, batch_size=2, device="cpu"):
        # self.model = MyNN(num_classes)
        self.model = model

        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        # self.optimizer = optim.SparseAdam(self.model.parameters(), lr=lrate)

        self.device = device
        # lrate = 0.01*0.75  # Adjust as needed
        momentum = 0.9  # Adjust as needed
        weight_decay = 0.0005  # Adjust as needed

        # Initialize the optimizer
        self.optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=momentum, weight_decay=weight_decay)
        print("optimizer: stochastic gradient descent ")

        self.batch_size = batch_size
        self.loss_function = nn.CrossEntropyLoss()

        print('lrate:', lrate)
        print('momentum:', momentum)

        if os.path.exists("logs"):
           shutil.rmtree('logs')

        self.writer = SummaryWriter('logs')  # 'logs' is the directory where TensorBoard will store the log files


    def train(self, out_model_path, train_loader, validation_loader, test_loader, num_epochs=2):

        self.saved_model_name = out_model_path
        self.best_model = None
        self.model.to(self.device)
        print('device', self.device)

        best_acc = 0
        timer = time_utils.timeMeasure()
        for epoch in range(num_epochs):
            # print("epoch", epoch, end="")

            timer.start()
            avg_loss = 0
            for data, target in tqdm(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)

                # Calculate SVM loss
                loss = self.loss_function(output, target)

                #loss = svm_loss(output, target) #we use SVM loss instead of CrossEntropyLoss
                avg_loss += loss.cpu().detach().numpy()

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

            if epoch % 3 == 0 and epoch > 0:
               acc_train = evaluator.evaluate(self.model, train_loader, self.device, dataset_name='train', images_list=None)
               acc_val = evaluator.evaluate(self.model, validation_loader, self.device, dataset_name='validation', images_list=None)
               
               self.writer.add_scalar('Train Acc', acc_train, global_step=epoch) #for tensorboard
               self.writer.add_scalar('Val Acc', acc_val, global_step=epoch) #for tensorboard
               
               if best_acc < acc_val:
                  print('saving when val acc is ', round(acc_val))

                  if out_model_path is not None:
                     torch.save(self.model.state_dict(), out_model_path) # model is saved
                      
                  self.best_model = copy.deepcopy(self.model)

                  print('model is saved...', end='')
                  best_acc = acc_val

               print("highest acc:", round(100 * best_acc, 2))
            print("epoch", str(epoch) + "of" + str(num_epochs))
            timer.end_new()

        # evaluate(best_model_copy, test_loader, device, dataset_name='best_model_copy test_loader', images_list=None)
        evaluator.evaluate(self.best_model, test_loader, self.device, dataset_name='test dataset', images_list=None)
        # evaluate(self.model, test_loader, device, dataset_name='self.model test_loader', images_list=None)
