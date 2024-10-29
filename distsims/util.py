import torch
import torch.nn.functional as F
from itertools import combinations


def cosine_similarity(models):
    similarities = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_similarities = []

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Compute cosine similarity between flattened parameters
            sim = F.cosine_similarity(param1_flat, param2_flat, dim=0)
            layer_similarities.append(sim.item())

        # Average similarity for this pair of models across all layers
        similarities.append(sum(layer_similarities) / len(layer_similarities))

    # Return the average similarity across all model pairs
    return sum(similarities) / len(similarities)


def euclidean_distance(models):
    distances = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_distances = []

        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if name1 != name2:
                raise ValueError("Model layers do not match.")
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Compute Euclidean distance between flattened parameters
            dist = torch.norm(param1_flat - param2_flat, p=2)
            layer_distances.append(dist.item())

            # if dist.item() > 0.0001:
            #     print(f"Layer '{name1}' divergence: {dist.item()}")

        # Average distance for this pair of models across all layers
        distances.append(sum(layer_distances) / len(layer_distances))

    # Return the average distance across all model pairs
    return sum(distances) / len(distances)


def mean_squared_difference(models):
    differences = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_differences = []

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Compute mean squared difference
            diff = torch.mean((param1_flat - param2_flat) ** 2)
            layer_differences.append(diff.item())

        # Average difference for this pair of models across all layers
        differences.append(sum(layer_differences) / len(layer_differences))

    # Return the average difference across all model pairs
    return sum(differences) / len(differences)


def parameter_correlation(models):
    correlations = []

    # Compare each pair of models
    for model1, model2 in combinations(models, 2):
        layer_correlations = []

        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param1_flat = param1.view(-1)
            param2_flat = param2.view(-1)

            # Compute Pearson correlation
            corr = torch.corrcoef(torch.stack([param1_flat, param2_flat]))[0, 1]
            layer_correlations.append(corr.item())

        # Average correlation for this pair of models across all layers
        correlations.append(sum(layer_correlations) / len(layer_correlations))

    # Return the average correlation across all model pairs
    return sum(correlations) / len(correlations)
