from Model import model
from DataLoader import DataLoader

mode = "TRAIN" # "TRAIN" or "VALIDATE"

class FilePaths:
    "filenames and paths for data"
    fnTrain = '' # to once training data can be found add here.
    
def main():
    "this is the main function it will kick everything off"
    loader = Dataloader(loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen))
    open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
    open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

    if mode == "TRAIN":
        model = Model(loader.charList, decoderType)
        train(model, loader)
    elif mode == "VALIDATE":
        model = Model(loader.charList, decoderType, mustRestore=True)
        validate(model, loader)

main()