# Loads a MobileNetV2 model pretrained on ImageNet via torchvision and replaces its
# final classification layer with a new linear head outputting 5 tomato leaf classes.
def build_mobilenetv2(num_classes: int = 5, pretrained: bool = True):
    pass