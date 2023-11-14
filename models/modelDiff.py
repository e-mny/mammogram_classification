import torch

def compare_models(model_paths):
    models = [torch.load(path)['net'] for path in model_paths]

    for i, model in enumerate(models):
        print(f"\nModel {i + 1} State Dict:")
        print(model.keys())

    for i in range(len(models) - 1):
        model1 = models[i]
        model2 = models[i + 1]

        # Compare state_dicts
        for key1, key2 in zip(model1.keys(), model2.keys()):
            if key1 != key2:
                print(f"\nDifference in key: {key1} (Model {i + 1}) vs {key2} (Model {i + 2})")

            param1 = model1[key1]
            param2 = model2[key2]

            if torch.equal(param1, param2):
                print(f"Parameters for key {key1} are equal.")
            else:
                print(f"Parameters for key {key1} are different.")

# Example usage
model_paths = ["/home/emok/sq58/Code/base_mammo/models/resnet50_224_1.pth", "/home/emok/sq58/Code/base_mammo/models/resnet50_224_2.pth", "/home/emok/sq58/Code/base_mammo/models/resnet50_224_3.pth"]
compare_models(model_paths)
