import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F


# Extract  pretrained activations
class SaveFeatures():
    """ Extract pretrained activations"""
    features=None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()
        
        
def camMap(model, data):
    final_layer = model._modules.get('Mixed_7c') # To change for every model
    activated_features = SaveFeatures(final_layer)

    ## Probabilities & labels for each images
    output = model(data[:8])# conver to cuda for softmax
    probabilities = F.softmax(output,dim=1).data.squeeze()
    pred_idx = np.argmax(probabilities.cpu().detach().numpy(),axis=1)
    labels = pred_idx
    activated_features.remove()
    print('Probabilities classes: %s \n Prediction indices %s \n Labels: %s' % (probabilities, pred_idx, labels))

    def getCAM(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv[0,:, :, ].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return cam_img

    weight_softmax_params = list(model._modules.get('fc').parameters()) # To change to classifier layer for every model
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    weight_softmax_params

    ## Current images & their heatmaps
    cur_images = data.cpu().numpy().transpose((0, 2, 3, 1))
    heatmaps = []
    for i in pred_idx:
        img = getCAM(activated_features.features, weight_softmax, i)
        heatmaps.append(img)
        
    print(cur_images.shape, len(heatmaps))

    # Probability for each images
    proba = []
    for i in probabilities.cpu().detach().numpy():
        idx = np.argmax(i)
        proba.append((str(np.round(i[idx]*100,2)))+'%')
    print(proba)

    fig=plt.figure(figsize=(20,15))
    for i in range(0, len(cur_images[:8])):
        img = cur_images[i]
        mask = heatmaps[i]
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        plt.imshow(img)
        plt.imshow(cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR), alpha=0.5, cmap='jet');
        ax.set_title('Label %d with %s probability' % (labels[i], proba[i]),fontsize=14)
        
    #cax = fig.add_axes([0.3, 0.42, 0.4, 0.04]) # place where be map
    cax = fig.add_axes([0.32, 0.42, 0.4, 0.03]) # place where be map
    clb = plt.colorbar(cax=cax, orientation='horizontal',ticks=[0, 0.5, 1])
    clb.ax.set_title('Level of "attention" NN in making prediction',fontsize=20)
    clb.ax.set_xticklabels(['low', 'medium', 'high'],fontsize=18)


    plt.show()