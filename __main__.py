from src.Classifier_pc import Classifier

# from src.Classifier import Classifier

classifier = Classifier(model="conversation", noise_lvl=10)
classifier.run()
