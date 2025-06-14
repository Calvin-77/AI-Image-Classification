import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm

def main():
    device = 'cuda'
    model_path = './vit_base-ckpt.t7'
    dataset_path = './dataset/real-vs-fake/test/'
    batch_size = 8
    image_size = 224

    print("==> Preparing test data..")
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5198, 0.4254, 0.3805), std=(0.2772, 0.2514, 0.2527)),
    ])

    testset = ImageFolder(root=dataset_path, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("==> Loading model..")
    checkpoint = torch.load(model_path)
    net = timm.create_model('vit_base_patch16_224', pretrained=False, drop_rate=0.1, drop_path_rate=0.2)
    net.head = torch.nn.Linear(net.head.in_features, 2)
    net.load_state_dict(checkpoint['model'])
    net = net.to(device)
    net.eval()

    print("==> Generating predictions..")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print("==> Calculating confusion matrix..")
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=testset.classes, yticklabels=testset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=testset.classes))

if __name__ == '__main__':
    main()
