from sklearn.neural_network import MLPRegressor

def get_autoencoder(hidden_layer_sizes=(32, 16)):
    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=200, random_state=42)

