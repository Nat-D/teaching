
class Logger():
    def __init__(self):
        self.accumulate_loss = 0

    def log_step(self, loss):
        self.accumulate_loss += loss


    def log_epoch(self, model):
        
        print(self.accumulate_loss)
        self.accumulate_loss = 0




