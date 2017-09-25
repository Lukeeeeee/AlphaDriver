from src.model.model import Model


class PreTrainer(Model):
    def __init__(self, config, sess_flag=False, data=None):
        super(PreTrainer, self).__init__(config, sess_flag, data)
