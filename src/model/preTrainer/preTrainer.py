from src.model.model import Model


class PreTrainer(Model):
    def __init__(self, config, data, pre_trained_model, sess_flag=False):
        super(PreTrainer, self).__init__(config, sess_flag, data)
        self.pre_trained_model = pre_trained_model

    def pre_train(self, *args, **kwargs):
        pass


