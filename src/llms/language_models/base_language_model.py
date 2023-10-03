import collections


class BaseLanguageModel(object):
    """
    Base lanuage model. Define how to generate sentence by using a LM
    Args:
        args: arguments for LM configuration
    """

    @staticmethod
    def add_args(parser):
        return

    def __init__(self, args):
        self.args = args

    def load_model(self, **kwargs):
        raise NotImplementedError

    def prepare_for_inference(self, **model_kwargs):
        raise NotImplementedError

    def tokenize(self, text):
        '''
        Return tokenized length of text

        Args:
            text (str): input text
        '''
        raise NotImplementedError

    def generate_sentence(self, lm_input):
        """
        Generate sentence by using a LM

        Args:
            lm_input (LMInput): input for LM
        """
        raise NotImplementedError
