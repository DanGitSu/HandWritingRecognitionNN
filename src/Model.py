import tensorflow as tf

class Model:

    batchSize = 50
    imgSize = {128, 32}
    maxTextLn = 32

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):

        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        self.setupCNN()
        self.setupRNN()
        self.setupCTC()