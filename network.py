from layer import Layer


class Network:
    layers = []
    layers_count = None

    def __init__(self, nodes, debug=False):
        layers_count = len(nodes)
        if layers_count < 2:
            print("Network must include at lease two layers")
            return

        print(f"Initializing network - nodes count: {layers_count}")

        # init layers
        layers = []
        for i in range(layers_count):
            if i == 0:
                layers.append(Layer(nodes[0], 0, Layer.Location.INPUT))
            elif i < layers_count:
                layers.append(Layer(nodes[i], nodes[i - 1], Layer.Location.HIDDEN))
            else:
                layers.append(Layer(nodes[i], nodes[i - 1], Layer.Location.OUTPUT))

        self.layers = layers
        self.layers_count = layers_count

        if debug:
            layers_info = [f"[{layer.input_count}, {layer.node_count}]" for layer in self.layers]
            print(f"Initialized network with layers: {', '.join(layers_info)}")

    def predict(self, inp):
        next_input = inp
        prediction = None
        for i in range(1, self.layers_count - 1):
            prediction = self.layers[i].forward_pass(next_input)
            next_input = prediction
        return prediction

    def predict_to_out(self, inp):
        next_input = inp
        prediction = None
        for i in range(1, self.layers_count - 1):
            prediction = self.layers[i].forward_pass(next_input)
            next_input = prediction
        return prediction

    def back_propagate(self, deltas):
        next_deltas = deltas
        for i in range(1, self.layers_count - 1):
            next_deltas = self.layers[self.layers_count - i - 1].backward_pass(next_deltas)
        return next_deltas

    def print_weights(self):
        for layer in self.layers:
            if layer.location is not Layer.Location.INPUT:
                print(layer.weights)

    def learn(self, inp, target):
        prediction = self.predict_to_out(inp)
        back_deltas = self.layers[self.layers_count - 1].output_round(prediction, target)
        self.back_propagate(back_deltas)
        self.print_weights()
