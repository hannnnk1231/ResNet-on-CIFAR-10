import torch
import matplotlib.pyplot as plt

class Training:
  def __init__(self, device, model, optimizer, criterion, scheduler = None, epochs = 100):
    self.device = device
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.scheduler = scheduler
    self.epochs = epochs
    self.train_acc_list = []
    self.val_acc_list = []
    self.train_loss_list = []
    self.val_loss_list = []
  
  def start_training(self, train_iterator, valid_iterator = None, logFileName = 'log.txt'):
    self.train_acc_list = []
    self.val_acc_list = []
    self.train_loss_list = []
    self.val_loss_list = []
    
    best_valid_loss = float('inf')

    with open(logFileName, "w")as f:
      for epoch in range(1, self.epochs+1):
        train_loss, train_acc = self.train(train_iterator)

        self.train_acc_list.append(train_acc)
        self.train_loss_list.append(train_loss)

        if valid_iterator:
          valid_loss, valid_acc = self.evaluate(valid_iterator)
          
          self.val_acc_list.append(valid_acc)
          self.val_loss_list.append(valid_loss)

          if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(self.model.state_dict(), 'best.pt')

          print('epoch:%d | Train Loss: %.03f | Train Acc: %.3f%% | Val Loss: %.03f | Val Acc: %.3f%%'
                        % (epoch, train_loss, train_acc*100, valid_loss, valid_acc*100))

          f.write('{},{},{},{},{}'.format(epoch, train_loss, train_acc, valid_loss, valid_acc))
          f.write('\n')
          f.flush()
        else:
          print('epoch:%d | Train Loss: %.03f | Train Acc: %.3f%%'
                        % (epoch, train_loss, train_acc*100))
          torch.save(self.model.state_dict(), 'best.pt')
          f.write('{},{},{}'.format(epoch, train_loss, train_acc))
          f.write('\n')
          f.flush()

        if self.scheduler != None:
          self.scheduler.step()

  def train(self, iterator):
      
      epoch_loss = 0
      epoch_acc = 0
      
      self.model.train()
      
      for (x, y) in iterator:
          
          x = x.to(self.device)
          y = y.to(self.device)
          
          self.optimizer.zero_grad()
                  
          y_pred = self.model(x)
          
          loss = self.criterion(y_pred, y)
          
          acc = self.calculate_accuracy(y_pred, y)
          
          loss.backward()
          
          self.optimizer.step()
          
          epoch_loss += loss.item()
          epoch_acc += acc.item()
          
      return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
  def evaluate(self, iterator):
      
      epoch_loss = 0
      epoch_acc = 0
      
      self.model.eval()
      
      with torch.no_grad():
          
          for (x, y) in iterator:

              x = x.to(self.device)
              y = y.to(self.device)

              y_pred = self.model(x)

              loss = self.criterion(y_pred, y)

              acc = self.calculate_accuracy(y_pred, y)

              epoch_loss += loss.item()
              epoch_acc += acc.item()
          
      return epoch_loss / len(iterator), epoch_acc / len(iterator)

  def test(self, iterator):
    self.model.load_state_dict(torch.load('best.pt'))
    test_loss, test_acc = self.evaluate(iterator)
    print('Test Loss: %.03f | Test Acc: %.3f%%' % (test_loss, test_acc*100))

  def calculate_accuracy(self, y_pred, y):
      top_pred = y_pred.argmax(1, keepdim = True)
      correct = top_pred.eq(y.view_as(top_pred)).sum()
      acc = correct.float() / y.shape[0]
      return acc

  def plot_acc(self):
    epochs = range(1, self.epochs+1)
    plt.plot(epochs, self.train_acc_list, 'bo', label="Training acc")
    plt.plot(epochs, self.val_acc_list, 'b', label="Validation acc")
    plt.legend()
    plt.show()

  def plot_loss(self):
    epochs = range(1, self.epochs+1)
    plt.plot(epochs, self.train_loss_list, 'ro', label="Training loss")
    plt.plot(epochs, self.val_loss_list, 'r', label="Validation loss")
    plt.legend()
    plt.show()