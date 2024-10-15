from models import SmallGenerativeTransformer


def get_layer_depth(model, depth=0):
    for child in model.children():
        print(f"Depth {depth}: Type {child.__class__.__name__}")
        get_layer_depth(child, depth + 1)


if __name__ == "__main__":
    model = SmallGenerativeTransformer(
        vocab_size=1000, embed_size=128, num_heads=4, num_layers=2
    )
    for depth, (name, module) in enumerate(model.named_modules()):
        print(f"Depth {depth}: Layer {name}, Type: {module.__class__.__name__}")
